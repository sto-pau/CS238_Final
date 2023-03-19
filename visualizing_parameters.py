"""
Run to test PyJulia is running as expected

Dependencies:
$ python3 -m pip install --user julia

$ python3
>>> import julia
>>> julia.install()

julia> using Pkg
julia> Pkg.add("PyCall")

Documentation:
https://pyjulia.readthedocs.io/_/downloads/en/latest/pdf/
"""
import numpy as np
import random

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("linear_model.jl")

import matplotlib.pyplot as plt

def plot_explore(s_1, s_2, s_3):
    # Use zip() to transpose the list of lists
    x1 , y1 = zip(*s_1)
    x2 , y2 = zip(*s_2)
    x3 , y3 = zip(*s_3)

    plt.scatter(x1,y1, label ='a=1')
    plt.scatter(x2,y2, label ='a=2')
    plt.scatter(x3,y3, label ='a=3')
    plt.legend()
    plt.show()

def dot_product(theta, s):
    return np.dot(theta, s)

def heatmap(theta, ax, fig):
    # Define the values of x and y
    x = np.linspace(0, 10, 100)
    y = np.linspace(-10, 10, 100)

    # Create a meshgrid of x and y values
    X, Y = np.meshgrid(x, y)

    # Create s with every combination of x and y
    s = np.row_stack((X.ravel(), Y.ravel()))

    # Compute the dot product of theta and s
    heat = dot_product(theta, s)

    # Reshape the heat array to match the shape of X and Y
    heat = heat.reshape(X.shape)

    # Plot the heatmap
    im = ax.pcolormesh(X, Y, heat, cmap='viridis')
    fig.colorbar(im, ax=ax)

def compare_actions(params):

    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    # Plot the heatmaps in each subplot
    heatmap(params[0], ax1, fig)
    heatmap(params[1], ax2, fig)
    heatmap(params[2], ax3, fig)

    plt.show()

if __name__ == '__main__':

    model = Main.create_model(2, 3)
    exploration_policy = Main.EpsilonGreedyExploration(0.3) #not used, need to get action

    s_1 = []
    s_2 = []
    s_3 = []

    for i in range(5000):
        max_y = 10
        y = np.random.uniform(low=-1, high=1) * max_y
        x = np.random.rand() * max_y

        #proxi_angle = abs(y)

        # if proxi_angle < 3:
        #     x = np.random.rand() * proxi_angle
        # else:
        #     x = np.random.rand() * max_y #if <5 x>y else >5 x<y

        s = [x, y]
        #sp = [0, 0]

        a = random.randint(1,3)

        if a == 1:
            s_1.append(s)
        elif a == 2:
            s_2.append(s)
        elif a == 3:
            s_3.append(s)

        if x > 3 and y > 3 and a == 3: #rotate CW
            r = 1
            sp = [x, y - 1]
        elif x > 3 and y < -3 and a == 1: #rotate CCW
            r = 1
            sp = [x, y + 1]
        elif abs(y) < 3 and x < 3: #stay center:
            if a == 2:
                r = 100
                sp = s
            # else:
            #     r = -100
            #     if a == 3:
            #         sp = [x+1, y - 1]
            #     elif a == 1:
            #         sp = [x+1, y + 1]
        else: #no reward for wrong action
            r = -10
            if a == 2: #don't move
                sp = s
            elif y < 0 and a == 3:
                sp = [x, y - 1]
            elif y > 0 and a == 1:
                sp = [x, y + 1]

        #print("s a r:", s, a, r)
        Main.update_b(model, s, a, r, sp)

    plot_explore(s_1, s_2, s_3)
    params = Main.get_params(model)

    compare_actions(params)

    for i in range(10):
        y = np.random.uniform(low=-1, high=1) * max_y
        x = np.random.rand() * max_y

        s = [x, y]
        print(s)
        print(Main.get_action(model, exploration_policy, s, False, True))