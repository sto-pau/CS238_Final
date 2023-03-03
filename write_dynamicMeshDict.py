import numpy as np
from numpy import linspace
import os

if __name__ == '__main__':
    path = '6Dof_test'

    # Use os.linesep to get the correct end-of-line character for the current OS
    eol = os.linesep

    total_steps = 2
    time_steps = np.array([0, 10])
    vel_x = np.array([0, 0])
    vel_y = np.array([0, 0])
    rotation = np.array([0, 0])

    line_list = ["{}{}{}({}".format(eol,total_steps,eol,eol)]
    line_list.extend([ "({} (({} {} 0) (0 0 {}))){}".format(time_steps[line], vel_x[line], vel_y[line], rotation[line], eol) for line in range(total_steps) ])
    line_list.extend([ "){}".format(eol) ])
    
    file_lines = ''.join(line_list)

    with open(path, 'w') as file:
        file.writelines(file_lines)

    file.close()