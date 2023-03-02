import numpy as np
from numpy import linspace

if __name__ == '__main__':
    path = '6Dof_test'

    total_steps = 3
    time_steps = linspace(0,total_steps,total_steps)
    vel_x = linspace(1,100,total_steps)
    vel_y = linspace(1,100,total_steps)
    rotation = linspace(1,np.pi,total_steps)

    line_list = [f"\n{total_steps}\n(\n"]
    line_list.extend([ f"({time_steps[line]} (({vel_x[line]} {vel_y[line]} 0) (0 0 {rotation[line]})))\n" for line in range(total_steps) ])
    line_list.extend([ f")\n" ])
    
    file_lines = ''.join(line_list)

    with open(path, 'w') as file:
        file.writelines(file_lines)

    file.close()