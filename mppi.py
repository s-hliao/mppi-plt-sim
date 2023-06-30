import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import math

class MPPI:
    def __init__(self, robot, state_dim, ctrl_dim, noise_mu, noise_sigma, u_min, u_max, dynamics,
                 running_cost, terminal_state_cost, 
                 expert_rollouts = None, expert_samples = None, expert_mu = None, expert_noise = None,
                 num_samples = 1000, horizon = 15, lambda_=1., sample_null_action= False,
                 timestep=1, device = "cpu"):
        self.robot = robot
        self.random_samples = num_samples
        if(expert_rollouts is None):
            self.K = num_samples
        else:
            if(expert_samples is None):
                self.K = num_samples + expert_rollouts.shape[0]
            else:
                self.K = num_samples + sum(expert_samples)
                
        self.T = horizon
        self.nx = state_dim # call this m
        self.nu = ctrl_dim # call this n
        self.timestep = timestep
        self.device = device
        self.sample_null_action = sample_null_action
        
        
        
        #tunable params:
        #number of samples, timesteps, control_min, control_max, 
        #dynamics, costmap
        #sampling distribution (noise_sigma), lambda (temperature) = free energy
        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = noise_sigma.to(self.device)
        self.dtype = noise_sigma.dtype
        self.noise_distr = MultivariateNormal(self.noise_mu, covariance_matrix = self.noise_sigma)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.lambda_ = lambda_ #larger temperatures enable more exploration
        
        self.u_min = torch.tensor(u_min,dtype=self.dtype, device=self.device)
        self.u_max = torch.tensor(u_max, dtype=self.dtype, device = self.device)
        
        self.u_init = torch.zeros(ctrl_dim, dtype=self.dtype, device = self.device)
        
        self.ctrl = self.noise_distr.sample((self.T, ))
        
        self.ctrl.to(self.device)
        
        self.expert_rollouts = expert_rollouts
        self.expert_samples = expert_samples

        self.expert_mu = expert_mu.to(self.device)
        if(self.expert_samples is not None and expert_noise is not None):
            self.expert_distr = []
            if len(expert_noise.shape) >=3:
                
                for i in range (expert_noise.shape[0]):
                    expert_sigma = expert_noise[i].to(self.device)
                    self.expert_distr.append(MultivariateNormal(self.expert_mu, covariance_matrix = expert_sigma))

            else:
                expert_sigma = expert_noise.to(self.device)
                self.expert_distr.append(MultivariateNormal(self.expert_mu, covariance_matrix = expert_sigma))
        
        self.total_cost = None
        self.total_cost_exponent = None
        self.omega = None
        self.states = None
        self.bounded_samples = None
        
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
    
    def find_control(self, state):
        self.ctrl = torch.roll(self.ctrl, -1, dims=0) #controls of shape (T, N) (t timesteps)
        
        self.ctrl[-1] = self.u_init #u_init is of shape (N)
        #roll controls back one timestep given that first one should have already been used
        
        if(not torch.is_tensor(state)):
            state = torch.tensor(state, dtype=self.dtype, device = self.device)
            
        self.cost_total = self.compute_total_cost(state) #S (Epsilon(k)) = total cost for each rollout 
        # shape: (K,)
        beta = torch.min(self.cost_total) #min total cost, 1 value
        self.total_cost_exponent = torch.exp(-1/self.lambda_ * (self.cost_total - beta)) 
        
        # exponent expression for all k
        eta = torch.sum(self.total_cost_exponent) #summation across all k of exponent, 1 value
        self.omega = (1./eta) * self.total_cost_exponent #weight omega(k) for each K, 1 value 
        #shape: (K,)
    
        perturbations = torch.sum(self.omega[:,None,None] * self.bounded_samples, dim=0) #size: (T,N)
        
        
        #omega is (K,), samples is (K,T, N), multiplication broadcasts to (K, N)
        #ends with summation of K, with t values in perturbations: (T, N)
        
        #add perturbations to controls as per algo for all T
        self.ctrl = self.ctrl+perturbations #shape : (T, N)

        del beta
        del eta
        del self.omega
        del state
        del perturbations

        
        return self.ctrl[0] # of size N
    
    def reset(self):
        self.ctrl = self.noise_distr.sample((self.T, ))
                                      
    def compute_total_cost(self, state):
        samples = self.noise_distr.rsample((self.random_samples, self.T)) # sample noise, shape: (K, T, N)
        perturbed_action = self.ctrl + samples # change actions by noise, 
        
        if(self.expert_rollouts != None):
            if(self.expert_samples != None):
                for i in range(self.expert_rollouts.shape[0]):
                    if(len(self.expert_distr) > 1):

                        samples = self.expert_distr[i].rsample((self.expert_samples[i],))
                        # expert rollouts[i] are K x T x N
                        # samples are 50 x TN 
                        perturbed_expert = torch.cat(
                            ((self.expert_rollouts[i, :, 0, None] + samples[:, :self.T, None]), 
                            (self.expert_rollouts[i, :, 1, None] + samples[:, self.T:, None])), axis = 2)

                    else:
                        samples = self.expert_distr[0].rsample((self.expert_samples[i], self.T))
                        perturbed_expert = self.expert_rollouts[i] + samples

                    perturbed_action = torch.cat((perturbed_action, perturbed_expert), axis = 0)
                    del perturbed_expert
            else:
                samples = self.noise_distr.rsample((self.random_samples, self.T))
                perturbed_action = torch.cat((perturbed_action, self.expert_rollouts), axis = 0)

        #shape: (T, N) + (K, T, N)
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        
        bounded_perturbed_action = self._bound_action(perturbed_action) # bound action limits
        if(self.expert_rollouts is not None): # debugging for expert rollouts
            bounded_expert_rollouts = self._bound_action(self.expert_rollouts)
        self.bounded_samples = bounded_perturbed_action-self.ctrl
        
        action_cost = self.lambda_ * torch.matmul(torch.abs(self.bounded_samples), self.noise_sigma_inv)
        perturbation_cost = torch.sum(self.ctrl * action_cost, dim=(1, 2))
        #matrix multiplication part over all T
        if(self.expert_rollouts is not None): #debugging for expert rollouts
            debug_cost, self.states, debug_actions= self.compute_rollout_costs(state, bounded_expert_rollouts)
            rollout_cost, placeholder, actions = self.compute_rollout_costs(state, bounded_perturbed_action)
        else:
            rollout_cost, self.states, actions = self.compute_rollout_costs(state, bounded_perturbed_action)
        
        #written as q(x_t) and phi(x_t) in the paper
        #states is Tx K x N
        self.cost_total = rollout_cost + perturbation_cost

        del samples
        del perturbed_action
        del action_cost
        del rollout_cost
        del perturbation_cost
        del bounded_perturbed_action

        return self.cost_total
                                            
    def compute_rollout_costs(self, state, perturbed_actions):
        if len(state.shape) is 1:
            state = torch.unsqueeze(state, 0)
            state = state.repeat(perturbed_actions.shape[0], 1)
        
        states = self.dynamics(state, perturbed_actions, self.timestep, dev = self.device)        
        horizon_rollout_costs = self.running_cost(states, perturbed_actions, self.timestep)

        rollout_costs = torch.sum(horizon_rollout_costs, axis = 1)
        rollout_costs += self.terminal_state_cost(states[:,-1,:]) # shape: (K,) , takes in last state
        
        return rollout_costs, states, perturbed_actions
                
    def _bound_action(self, action):
        bounded_action = torch.empty_like(action, dtype=self.dtype, device = self.device)
            
        if self.u_max != None and self.u_min != None:
            max_compare = torch.ones_like(action, dtype=self.dtype, device = self.device) * self.u_max
            min_compare = torch.ones_like(action, dtype=self.dtype, device = self.device) * self.u_min
            bounded_action = torch.max(torch.min(action, max_compare), min_compare)

            del max_compare
            del min_compare
                
            return bounded_action
        return action


class MPPI_path_follower:
    def __init__(self, robot, state_dim, ctrl_dim, noise_mu, noise_sigma, u_min, u_max, dynamics,
                 env, running_cost, terminal_state_cost, path, 
                 num_samples = 1000, horizon = 15, lambda_=1., sample_null_action= False,
                 timestep=1, device = "cpu"):
        self.robot = robot
        self.K = num_samples
        self.T = horizon
        self.nx = state_dim # call this m
        self.nu = ctrl_dim # call this n
        self.timestep = timestep
        self.device = device
        self.sample_null_action = sample_null_action
        self.env = env
        
        #tunable params:
        #number of samples, timesteps, control_min, control_max, 
        #dynamics, costmap
        #sampling distribution (noise_sigma), lambda (temperature) = free energy
        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = noise_sigma.to(self.device)
        self.dtype = noise_sigma.dtype
        self.noise_distr = MultivariateNormal(self.noise_mu, covariance_matrix = self.noise_sigma)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.lambda_ = lambda_ #larger temperatures enable more exploration
        
        self.u_min = torch.tensor(u_min,dtype=self.dtype, device=self.device)
        self.u_max = torch.tensor(u_max, dtype=self.dtype, device = self.device)
        
        self.u_init = torch.zeros(ctrl_dim, dtype=self.dtype, device = self.device)
        
        self.ctrl = self.noise_distr.sample((self.T, ))
        
        self.ctrl.to(self.device)
        
        
        
        self.total_cost = None
        self.total_cost_exponent = None
        self.omega = None
        self.states = None
        self.bounded_samples = None
        
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        
        
        self.path = path        
        self.following_index = 0 
    
    def find_control(self, state):
        
        waypoint = self.path[self.following_index]
        dist_to_goal =math.sqrt((self.robot.x-waypoint[0]) * (self.robot.x-waypoint[0]) + (self.robot.y-waypoint[1]) * (self.robot.y-waypoint[1]))
        
        if(dist_to_goal <= 9):
            next_waypoint = 0
            for i in range(1,10):
                if(self.following_index+i < len(self.path)):
                    future = self.path[self.following_index+i]
                    dist_to_next = math.sqrt((self.robot.x-future[0]) * (self.robot.x-future[0]) + (self.robot.y-future[1]) * (self.robot.y-future[1]))
                    if dist_to_next < 10:
                        next_waypoint = i
            self.following_index += next_waypoint
            
            x_goal, y_goal, run_cost, term_cost = self.env.goal_point
            self.env.goal_point =self.path[self.following_index][0], self.path[self.following_index][1], run_cost, term_cost
        
            
        self.ctrl = torch.roll(self.ctrl, -1, dims=0) #controls of shape (T, N) (t timesteps)
        
        self.ctrl[-1] = self.u_init #u_init is of shape (N)
        #roll controls back one timestep given that first one should have already been used
        
        if(not torch.is_tensor(state)):
            state = torch.tensor(state, dtype=self.dtype, device = self.device)
            
        self.cost_total = self.compute_total_cost(state) #S (Epsilon(k)) = total cost for each rollout 
        # shape: (K,)
        beta = torch.min(self.cost_total) #min total cost, 1 value
        self.total_cost_exponent = torch.exp(-1/self.lambda_ * (self.cost_total - beta)) 
        
        # exponent expression for all k
        eta = torch.sum(self.total_cost_exponent) #summation across all k of exponent, 1 value
        self.omega = (1./eta) * self.total_cost_exponent #weight omega(k) for each K, 1 value 
        #shape: (K,)
    
        perturbations = torch.sum(self.omega[:,None,None] * self.bounded_samples, dim=0) #size: (T,N)
        
        
        #omega is (K,), samples is (K,T, N), multiplication broadcasts to (K, N)
        #ends with summation of K, with t values in perturbations: (T, N)
        
        #add perturbations to controls as per algo for all T
        self.ctrl = self.ctrl+perturbations #shape : (T, N)

        del beta
        del eta
        del self.omega
        del state
        del perturbations


        
        return self.ctrl[0] # of size N
    
    def reset(self):
        self.ctrl = self.noise_distr.sample((self.T, ))
                                      
    def compute_total_cost(self, state):
        samples = self.noise_distr.rsample((self.K, self.T)) # sample noise, shape: (K, T, N)
        perturbed_action = self.ctrl + samples # change actions by noise, 
        #shape: (T, N) + (K, T, N)
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
            
        bounded_perturbed_action = self._bound_action(perturbed_action) # bound action limits
        self.bounded_samples = bounded_perturbed_action-self.ctrl
        
        action_cost = self.lambda_ * torch.matmul(torch.abs(self.bounded_samples), self.noise_sigma_inv)
        perturbation_cost = torch.sum(self.ctrl * action_cost, dim=(1, 2))
        #matrix multiplication part over all T
        
                                            
        rollout_cost, self.states, actions = self.compute_rollout_costs(state, bounded_perturbed_action)
        #written as q(x_t) and phi(x_t) in the paper
        #states is Tx K x N
        self.cost_total = rollout_cost + perturbation_cost

        del samples
        del perturbed_action
        del action_cost
        del rollout_cost
        del perturbation_cost

        return self.cost_total
                                            
    def compute_rollout_costs(self, state, perturbed_actions):
        if len(state.shape) is 1:
            state = torch.unsqueeze(state, 0)
            state = state.repeat(self.K, 1)
            
        states = self.dynamics(state, perturbed_actions, self.timestep, dev = self.device)
        horizon_rollout_costs = self.running_cost(states, perturbed_actions, self.timestep)
        
        rollout_costs = torch.sum(horizon_rollout_costs, axis = 1)
        rollout_costs += self.terminal_state_cost(states[:,-1,:]) # shape: (K,) , takes in last state
        
        return rollout_costs, states, perturbed_actions
                
    def _bound_action(self, action):
        bounded_action = torch.empty_like(action, dtype=self.dtype, device = self.device)
            
        if self.u_max != None and self.u_min != None:
            max_compare = torch.ones_like(action, dtype=self.dtype, device = self.device) * self.u_max
            min_compare = torch.ones_like(action, dtype=self.dtype, device = self.device) * self.u_min
            bounded_action = torch.max(torch.min(action, max_compare), min_compare)

            del max_compare
            del min_compare
                
            return bounded_action
        return action