
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from queue import PriorityQueue
import math

class a_star_planner:
    def __init__(self, robot, state_dim, ctrl_dim, u_min, u_max, costmap_func, heuristic_func,
                 iterations,goal_point, goal_tolerance, grid_interval=0.2,
                 timestep=1, angle_density = 32,  device = "cpu"):
        self.robot = robot
        self.nx = state_dim
        self.nu = ctrl_dim
        self.u_min = u_min
        self.u_max = u_max
        
        self.costmap_func = costmap_func
        self.heuristic_func = heuristic_func
        
        self.iterations = iterations
        self.goal_point = goal_point
        self.goal_tolerance = goal_tolerance
        
        self.device = device
        self.angle_density = angle_density
        self.scale = grid_interval
        self.explored = []
        dim = (int)(100/self.scale) + 1
        
        self.timestep = timestep
        
        axis = torch.ones(dim, device = self.device)* self.scale
        axis = torch.cumsum(axis, axis =0)-self.scale
        
        
        theta = torch.ones(angle_density, device = self.device) * 2*math.pi/angle_density
        theta = torch.cumsum(theta, axis =0)-2*math.pi/angle_density
        steer = torch.zeros(angle_density, device = self.device) * 2*math.pi/angle_density
        
        states = torch.empty([dim, dim, angle_density, state_dim], device = self.device)
        states[:,:, :, 0] = axis[:, None, None]
        states[:,:, :, 1] = axis[None, :, None]
        states[:,:, :, 2] = theta[None, None, :]
        states[:,:, :, 3] = steer[None, None, :]
        
        
        linearized_states = torch.reshape(states,(dim*dim*angle_density, -1))
        linearized_h = self.heuristic_func(linearized_states)
        linearized_costmap = self.costmap_func(linearized_states) + (self.timestep) #penalizes extra movements
        self.h = torch.reshape(linearized_h, (dim, dim, angle_density))
        self.costmap = self.timestep + torch.reshape(linearized_costmap, (dim, dim, angle_density))
        
        self.cost = torch.zeros([dim, dim, angle_density], device = self.device)
        
        (x, y, angle_theta, steer) = self.robot.get_state()
        theta = int (angle_theta/(math.pi * 2) *angle_density)


        goal_x, goal_y, run_cost, term_cost = self.goal_point
        
        startstate = (int(x/self.scale), int(y/self.scale), theta)
        curstate = (int(x/self.scale), int(y/self.scale), theta, steer)
        parent = torch.ones([dim, dim, angle_density, 3], device = self.device, dtype=torch.long)*-1
        
        q = PriorityQueue()
        
        q.put((0, curstate))
        
        
        iteration_count = 0
        
        self.path_end = None
        
        
        while not q.empty() and iteration_count < self.iterations:
            heuristic_cost, curstate = q.get()
            x, y, theta, steer = curstate
            
            neighbors = []    
            
            speed = 0
            angle_change = 0
            
            for angle_change in range(-2, 3):
                for i in range (3,10):
                    speed = i * 0.5
                    actual_pos_x = x * self.scale
                    actual_pos_y = y * self.scale
                    
                    discrete_theta = (theta + angle_change+ angle_density) % angle_density
                    
                    next_theta = discrete_theta * 2 * math.pi / angle_density
                    
                    next_pos_x = actual_pos_x + (speed * math.cos(next_theta) * self.timestep)
                    next_pos_y = actual_pos_y + (speed * math.sin(next_theta) * self.timestep)
                    
                    
                    discrete_x = int(round(next_pos_x/self.scale))
                    discrete_y = int(round(next_pos_y/self.scale))
                    
                    if(discrete_x > 0 and discrete_y > 0 
                       and discrete_x < 100/self.scale and discrete_y < 100/self.scale):
                        neighbors.append((discrete_x, discrete_y, discrete_theta))
            
            dist = math.sqrt((x*self.scale-goal_x)*(x*self.scale-goal_x) 
                             + (y*self.scale-goal_y)*(y*self.scale-goal_y))
            if dist < self.goal_tolerance:
                self.path_end = torch.tensor((x, y, theta),dtype = torch.long, device = self.device)
            
            for next_x, next_y, next_theta in neighbors:
                
                next_cost = self.cost[x, y, theta] + self.costmap[next_x, next_y, next_theta]
                if self.cost[next_x, next_y, next_theta]==0 or next_cost < self.cost[next_x, next_y, next_theta]:
                    self.cost[next_x, next_y, next_theta] = next_cost
                    
                    next_heuristic_cost = self.cost[next_x, next_y, next_theta] + self.h[next_x, next_y, next_theta]
                    
                    parent[next_x, next_y, next_theta,:] = torch.tensor([x, y, theta],device = self.device)
                    if(self.path_end is None):
                        q.put((next_heuristic_cost.item(), (next_x, next_y, next_theta, 0)))
                        
            self.explored.append([x*self.scale, y*self.scale])
                
            #if(iteration_count %1000==0):
                #print(iteration_count, heuristic_cost)
                #print(x, y, theta)
            #iteration_count+=1
            
                
        
        self.path=[]
        if(self.path_end is not None):
            curstate = self.path_end
            self.path.append((
                    curstate[0].item()*self.scale,
                    curstate[1].item()*self.scale,
                    curstate[2].item()*2*math.pi/self.angle_density,
                    0
                ))
            
            while not torch.all(curstate == 
                                torch.tensor(startstate, device = self.device, dtype = torch.long)):

                curstate = parent[curstate[0], curstate[1], curstate[2]]
                self.path.append((
                    curstate[0].item()*self.scale,
                    curstate[1].item()*self.scale,
                    curstate[2].item()*2*math.pi/self.angle_density,
                    0
                ))
            self.path.reverse()
                


            
                
        
        
        
    