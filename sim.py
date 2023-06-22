from skimage import io
import os
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
import math
import draw_utils

class Simulation:
    def __init__(self, robot, controller, env, size=(100,100), timestep=1, goal_tolerance = 0.1):
        self.robot = robot
        self.x_size = size[0]
        self.y_size = size[1]
        self.timestep = timestep
        self.steps = 0
        self.env = env
        self.save_snapshot()
        self.steps+=1
        self.controller = controller
        
        self.goal_tolerance = goal_tolerance
        
    def run(self, iterations=10, write_snapshots = True, draw_obstacles = True,
            write_rollouts=False, write_controls=False, write_rollout_start = 0,
            write_rollouts_num= 1, write_rate = 5):
        for step in range(iterations):
            velocity, steer_rate = self.controller.find_control(self.robot.get_state())
            if(write_snapshots and self.steps % write_rate==0):
                rollouts = None
                controls = None
                if(write_rollouts):
                    rollouts = self.controller.states

                if(write_controls):
                    controls = self.controller.ctrl 
                
                self.save_snapshot(projected_rollouts=rollouts, draw_obstacles=draw_obstacles,
                               selected_controls=controls, write_start = write_rollout_start,
                               n_rollouts = write_rollouts_num)
                

            self.robot.update(velocity, steer_rate, self.timestep)
            self.steps+=1
            
            x_dist = (self.robot.x-self.env.goal_point[0])
            y_dist = (self.robot.y-self.env.goal_point[1])
            dist = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
            
            if(dist < self.goal_tolerance):
                self.save_snapshot(full_history = True)
                return step, True
        self.save_snapshot(full_history = True)
        return iterations, False
    
    
    def draw_explored(self, points, draw_angle=True):
        img = io.imread('grid_images/grid_small.jpg')
        fig = px.imshow(img, color_continuous_scale = "greys", 
                        origin = "lower",width=800, height=800)
        
        draw_utils.draw_points(fig, self.env.goal_point, self.env.avoidance_points)
        
        draw_utils.draw_rect_obstacles(fig, self.env.rect_obstacles)
        
        draw_utils.draw_circle_obstacles(fig, self.env.circle_obstacles)
        
       
        
        fig.add_scatter(x = [coord[0] for coord in points],
                   y = [coord[1] for coord in points],
                   mode = "markers",
                   marker =dict(color="green", opacity=1, size=10))
        if draw_angle:
            draw_utils.draw_arrow_annotations(fig, points)
               
        fig.update_layout(showlegend=False)
        fig.update_layout(coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(range=[0, 100], autorange=False, scaleratio=1)
        fig.update_yaxes(range=[0, 100], autorange=False, scaleratio=1)
        fig.show()
        
        
        
        
        
        
    def save_snapshot(self, draw_obstacles = True, projected_rollouts=None, selected_controls=None, write_start = 0, n_rollouts = 1, full_history=False):
        
        img = io.imread('grid_images/grid_small.jpg')
        fig = px.imshow(img, color_continuous_scale = "greys", 
                        origin = "lower",width=800, height=800)
        
        
        
        # draw annotations
        if full_history:
            history = self.robot.get_history()
            fig.add_scatter(x = [coord[0] for coord in history],
                       y = [coord[1] for coord in history],
                       mode = "markers",
                       marker =dict(color="green", opacity=1, size=10))
            
            draw_utils.draw_arrow_annotations(fig, self.robot.history)

        else:
            fig.add_scatter(x = [self.robot.history[-1][0]],
                       y = [self.robot.history[-1][1]],
                       mode = "markers",
                       marker =dict(color="red", opacity=1, size=10))
            
            if(projected_rollouts != None):
                horizon = len(projected_rollouts)
                x_coords = torch.Tensor.cpu(projected_rollouts[write_start:write_start+n_rollouts, :,0])
                y_coords = torch.Tensor.cpu(projected_rollouts[write_start:write_start+n_rollouts, :,1])
                for rollout in range(n_rollouts):
                     fig.add_trace(go.Scatter(x = x_coords[rollout],
                       y = y_coords[rollout],
                       mode = "lines+markers",
                       marker =dict(color="orange", opacity=1, size=10)))
                    
                    
            if(selected_controls != None):
                selected_rollout = self.robot.get_rollout_actions(selected_controls, self.timestep)
                fig.add_trace(go.Scatter(x = [coord[0] for coord in selected_rollout],
                       y = [coord[1] for coord in selected_rollout],
                       mode = "lines+markers",
                       marker =dict(color="green", opacity=1, size=10)))
        
            draw_utils.draw_arrow_annotations(fig, [self.robot.history[-1]])


        
        if (draw_obstacles):
            draw_utils.draw_points(fig, self.env.goal_point, self.env.avoidance_points)
            draw_utils.draw_rect_obstacles(fig, self.env.rect_obstacles)
            
            draw_utils.draw_circle_obstacles(fig, self.env.circle_obstacles)
            
        fig.update_layout(showlegend=False)
        fig.update_layout(coloraxis_showscale=False,
                      margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(range=[0, 100], autorange=False, scaleratio=1)
        fig.update_yaxes(range=[0, 100], autorange=False, scaleratio=1)
        if not os.path.exists("sim_images"):
            os.mkdir("sim_images")
        
        if full_history:
            fig.write_image("sim_images/sim_full.png")
        else:
            
            fig.write_image("sim_images/sim_"+str(self.steps)+".png")
            
    def display_run(self):
        img_list = []
        k = 1
        
        if not os.path.exists("sim_images"):
            return
        for step in range(self.steps):
            if os.path.isfile("sim_images/sim_"+str(step)+".png"):
                img = io.imread("sim_images/sim_"+str(step)+".png")
                img_list.append(img)
                os.remove("sim_images/sim_"+str(step)+".png")

        # If no images, don't do anything
        if len(img_list) == 0:
            return

        img_list = np.array(img_list)
        fig = px.imshow(img_list, animation_frame=0)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()
        
    def display_history(self):
        if not os.path.exists("sim_images"):
            return
        
        img = io.imread("sim_images/sim_full.png")    

        fig = px.imshow(img)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.show()
        
        
    