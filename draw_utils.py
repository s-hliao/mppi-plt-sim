import math

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def draw_points(fig, goal_point, avoidance_points=[] ):
    goal_2_x0 = goal_point[0]-2
    goal_2_y0 = goal_point[1]-2
    goal_2_x1 = goal_point[0]+2
    goal_2_y1 = goal_point[1]+2

    fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=goal_2_x0, y0=goal_2_y0,
            x1=goal_2_x1, y1=goal_2_y1,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="Yellow",opacity = 0.75,
        )

    goal_5_x0 = goal_point[0]-5
    goal_5_y0 = goal_point[1]-5
    goal_5_x1 = goal_point[0]+5
    goal_5_y1 = goal_point[1]+5

    fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=goal_5_x0, y0=goal_5_y0,
            x1=goal_5_x1, y1=goal_5_y1,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="Yellow",opacity = 0.5,
        )

    goal_10_x0 = goal_point[0]-10
    goal_10_y0 = goal_point[1]-10
    goal_10_x1 = goal_point[0]+10
    goal_10_y1 = goal_point[1]+10

    fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=goal_10_x0, y0=goal_10_y0,
            x1=goal_10_x1, y1=goal_10_y1,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="Yellow",opacity = 0.25,
        )
    
    for x, y, weight in avoidance_points:
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x-2*weight, y0=y-2*weight,
            x1=x+2*weight, y1=y+2*weight,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="Blue",opacity = 0.75,
        )
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x-5*weight, y0=y-5*weight,
            x1=x+5*weight, y1=y+5*weight,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="Blue",opacity = 0.5
        )
        
    
def draw_arrow_annotations(fig, robot_history):
    arrow_list = []
    for x1, y1, theta in robot_history:
        arrow = go.layout.Annotation(dict(
                        x=x1+(5*math.cos(theta)),
                        y=y1+(5*math.sin(theta)),
                        xref="x", yref="y",
                        text="",
                        showarrow=True,
                        axref="x", ayref='y',
                        ax=x1,
                        ay=y1,
                        arrowhead=3,
                        arrowwidth=2,
                        arrowcolor='red',)
                    )
        arrow_list.append(arrow)
        
    fig.update_layout(annotations=arrow_list)

def draw_circle_obstacles(fig, obstacles):
    for x, y, radius in obstacles:
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x-radius, y0=y-radius,
            x1=x+radius, y1=y+radius,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="White",opacity = 1.0,
        )
    
def draw_rect_obstacles(fig, obstacles):
    for x_0, y_0, x_1, y_1 in obstacles:
        fig.add_shape(type="rect",
            xref="x", yref="y",
            x0=x_0, y0=y_0,
            x1=x_1, y1=y_1,
            line=dict(
                color="RoyalBlue",
                width=0,
            ),
            fillcolor="White",opacity = 1.0,
        )
        