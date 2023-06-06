import numpy as np
import torch

import math

class Bicycle:
    def __init__(self,x, y,theta, L):
        self.x = x
        self.y = y
        self.theta = theta
        self.steer = 0
        self.L = 0
        self.history = [(x,y,theta,0)]
        self.control_history = []
        
    def get_state(self):
        return self.x, self.y, self.theta, self.steer
    
    def get_rollout_actions(self, actions, delta_t):
        
        
        curstate = (self.x, self.y, self.theta, self.steer)
        states = [curstate]
        for action in actions:
            velocity=action[0]
            steer_rate = action[1]
            x, y, theta, steer = curstate
            x_dot = velocity*math.cos(theta) 
            y_dot = velocity*math.sin(theta) 
            theta_dot = steer
            if(self.L is not 0):
                theta_dot = velocity/(self.L/math.tan(steer))

            x+= x_dot * delta_t
            y += y_dot * delta_t
            theta+= theta_dot * delta_t
            steer+= steer_rate * delta_t

            if(torch.is_tensor(x)):
                x = x.item()
            if(torch.is_tensor(y)):
                y = y.item()
            if(torch.is_tensor(theta)):
                theta = theta.item()
            if(torch.is_tensor(steer)):
                steer = steer.item()
        
            states.append((x, y, theta, steer))
            curstate = (x, y, theta, steer)
            
        return states
    
    
    def dynamics_batch_horizon(self, state, actions, delta_t, dev="cpu"): 
        x = torch.unsqueeze(state[:,0], 1)
        y = torch.unsqueeze(state[:,1], 1)
        theta = torch.unsqueeze(state[:,2], 1)
        steer = torch.unsqueeze(state[:,3], 1)
        # batchified for many controls and states, overall timesteps
        velocity = actions[:,:,0]
        steer_rate = actions[:, :, 1]      
            
        horizon_states = torch.empty((actions.shape[0], actions.shape[1],
                                          state.shape[1]), dtype=state.dtype, device =dev)
        
        steer_dot = steer_rate #steer_dot
        steer_displacement = torch.cumsum(steer_dot * delta_t, axis=1) # sum across all t
        horizon_states[:,:,3] = steer + steer_displacement # all steer positions across the horizon
        
        
        if(self.L is not 0):
            theta_dot  = velocity/(self.L/torch.tan(steer))
        else:
            theta_dot = horizon_states[:,:,3] #theta_dot
            
        theta_displacement = torch.cumsum(theta_dot * delta_t, axis=1) # sum across all t
        horizon_states[:,:,2] = theta + theta_displacement # all theta positions across the horizon
        
        x_dot = velocity*torch.cos(horizon_states[:,:,2]) # x_dot based on theta
        x_displacement = torch.cumsum(x_dot * delta_t, axis=1) # sum across all t
        horizon_states[:,:,0] = x +x_displacement # all x positions across the horizon
        
        y_dot =  velocity*torch.sin(horizon_states[:,:,2]) # y_dot based on theta
        y_displacement = torch.cumsum(y_dot * delta_t, axis=1) # sum across all t
        horizon_states[:,:,1] = y+y_displacement # all x positions across the horizon
        
        # horizon_states should be of shape : K x M

        return horizon_states
    
    def update(self, velocity, steer_rate, delta_t):
        x_dot = velocity*math.cos(self.theta) 
        y_dot = velocity*math.sin(self.theta) 
        theta_dot = self.steer
        if(self.L is not 0):
            theta_dot = velocity/(self.L/math.tan(self.steer))
        
        self.x+= x_dot * delta_t
        self.y += y_dot * delta_t
        self.theta+= theta_dot * delta_t
        self.steer+= steer_rate * delta_t
        
        if(torch.is_tensor(self.x)):
            self.x = self.x.item()
        if(torch.is_tensor(self.y)):
            self.y = self.y.item()
        if(torch.is_tensor(self.theta)):
            self.theta = self.theta.item()
        if(torch.is_tensor(self.steer)):
            self.steer = self.steer.item()
        
        self.history.append((self.x, self.y, self.theta, self.steer))
        self.control_history.append(torch.tensor([velocity, steer_rate], dtype =torch.float))
        
        
    def get_history(self):
        return self.history
        
        
class Map:# may have to instantiate map class for more complex costmaps and stuff
    def __init__(self, goal_point, avoidance_points =[], rect_obstacles=[], circle_obstacles=[], device="cpu", obstacle_penalty=1000000.):
        self.goal_point = goal_point
        self.avoidance_points = avoidance_points
        self.rect_obstacles = rect_obstacles
        self.circle_obstacles = circle_obstacles
        self.device = device
        self.obstacle_penalty = obstacle_penalty
        
    def running_cost_batch_horizon(self, states, actions, dt):
        #batchified for many time points, states and controls
        x = states[:,:,0]
        y = states[:,:,1]
        theta = states[:,:,2]
        steer = states[:,:,3]
        velocity = actions[:,:,0]
        steer_rate = actions[:,:,1]
        
        #print(self.goal_point)
        
        x_goal, y_goal, run_cost, term_cost = self.goal_point
        
        dist_to_goal = run_cost * torch.sqrt(((x-x_goal) * (x-x_goal)) + ((y-y_goal) * (y-y_goal)))
        
        in_obstacle_cost = torch.zeros_like(dist_to_goal, device = self.device)
        avoid_cost = torch.zeros_like(dist_to_goal, device = self.device)
        
        for x_avoid, y_avoid, weight in self.avoidance_points:
            dist = torch.sqrt(((x-x_goal) * (x-x_goal)) + ((y-y_goal) * (y-y_goal)))
            
            in_range = dist.le(20)
            mask = torch.zeros_like(avoid_cost, device =self.device)
            
            mask[in_range] = 1
            mask = mask * dist
            
            avoid_cost += weight/(0.5+mask)
            
        rect_mask = torch.zeros_like(avoid_cost, device =self.device)==1
        for x0, y0, x1, y1 in self.rect_obstacles:
            
            mask1=x.ge(x0)  
            
            mask2 = x.le(x1)
        
            mask3 = y.ge(y0)
            
            mask4 = y.le(y1) 
            in_map= x.ge(0) & x.le(100) & y.ge(0) & y.le(100)
            
            rect_mask = rect_mask |(mask1 & mask2 & mask3 & mask4) | torch.logical_not(in_map)
            
            
        
        circle_mask = torch.zeros_like(avoid_cost, device =self.device)==1
        for x0, y0, radius in self.circle_obstacles:
            dist = torch.sqrt(((x-x0) * (x-x0)) + ((y-y0) * (y-y0)))
            
            inside_circle = dist.le(radius)  
            
            circle_mask = circle_mask | inside_circle
            
        rollouts_in_obstacle = (circle_mask + rect_mask).sum(axis = 1).ge(1)
        in_obstacle_cost[rollouts_in_obstacle==True] = self.obstacle_penalty
        
        return torch.max(dist_to_goal + avoid_cost, in_obstacle_cost)
    
    def terminal_state_cost_batch(self, state):         
             
        
        return torch.max(self.get_distance_batch(state), self.get_obstacles_batch(state))
    
    def get_distance_batch(self, state):
        x = state[:,0]
        y = state[:,1]
        theta = state[:,2]
        steer = state[:,3]
        
        x_goal, y_goal, run_cost, term_cost = self.goal_point
        
        dist_to_goal = term_cost * torch.sqrt(((x-x_goal) * (x-x_goal)) + ((y-y_goal) * (y-y_goal)))
                                              
        avoid_cost = torch.zeros_like(dist_to_goal, device = self.device)
        
        for x_avoid, y_avoid, weight in self.avoidance_points:
            dist = torch.sqrt(((x-x_goal) * (x-x_goal)) + ((y-y_goal) * (y-y_goal)))
            
            in_range = dist.le(20)
            mask = torch.zeros_like(avoid_cost, device =self.device)
            
            mask[in_range] = 1
            mask = mask * dist
            
            avoid_cost += weight/(1+mask)
        
        return dist_to_goal + avoid_cost
                                              
    def get_obstacles_batch(self, state):
        x = state[:,0]
        y = state[:,1]
        theta = state[:,2]
        steer = state[:,3]
                                              
        rect_mask = torch.zeros_like(x, device =self.device)==1
        for x0, y0, x1, y1 in self.rect_obstacles:
            
            mask1=x.ge(x0)  
            
            mask2 = x.le(x1)
        
            mask3 = y.ge(y0)
            
            mask4 = y.le(y1) 
            
            in_map= x.ge(0) & x.le(100) & y.ge(0) & y.le(100)
            
            rect_mask = rect_mask |(mask1 & mask2 & mask3 & mask4) | torch.logical_not(in_map)
        
        circle_mask = torch.zeros_like(rect_mask, device =self.device)==1
        for x0, y0, radius in self.circle_obstacles:
            dist = torch.sqrt(((x-x0) * (x-x0)) + ((y-y0) * (y-y0)))
            
            inside_circle = dist.le(radius)  
            
            circle_mask = circle_mask | inside_circle
            
        in_obstacle_cost = self.obstacle_penalty * (circle_mask + rect_mask).ge(1)
                                              
        return in_obstacle_cost
                                              
                                              
    