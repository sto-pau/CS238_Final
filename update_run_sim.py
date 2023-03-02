import numpy as np
from numpy import linspace
import subprocess
import os

def controlDictupdate(new_start_time, new_end_time, new_writeInterval, case_name):
    eol = os.linesep
    path = case_name + '/backGround/system/controlDict'

    with open(path, 'r') as file:
        file_lines = file.readlines()

    file.close()

    for line_number , line_content in enumerate(file_lines):
        if line_content.startswith("startTime"):
            file_lines[line_number] = "startTime       {};{}".format(new_start_time, eol) #7 spaces
        elif line_content.startswith("endTime"):
            file_lines[line_number] = "endTime         {};{}".format(new_end_time, eol) #9 spaces
        elif line_content.startswith("writeInterval"):
            file_lines[line_number] = "writeInterval   {};{}".format(new_writeInterval, eol) #3 spaces

    with open(path, 'w') as file:
        file.writelines(file_lines)

    file.close()

def dynamicMeshDictupdate(total_steps, time_steps, vel_x, vel_y, rotation, case_name):
    eol = os.linesep
    path = case_name + '/backGround/constant/6DoF.dat'

    line_list = ["{}{}{}({}".format(eol,total_steps,eol,eol)]
    line_list.extend([ "({} (({} {} 0) (0 0 {}))){}".format(time_steps[line], vel_x[line], vel_y[line], rotation[line], eol) for line in range(total_steps) ])
    line_list.extend([ "){}".format(eol) ])
    
    file_lines = ''.join(line_list)

    with open(path, 'w') as file:
        file.writelines(file_lines)

    file.close()

def runSim(case_name, runMesh = False):

    if runMesh == True:
        path_mesh = case_name + '/run_mesh.sh'
        subprocess.run([path_mesh])

    path_submit = case_name + '/backGround/submit_run.sh'    
    subprocess.run(['sbatch', path_submit])

if __name__ == '__main__':
    
    #filepaths to sim
    case_name = 'CS238_dummy_case'

    #paramters for running simulation duration
    new_start_time = 0.0 #0
    new_end_time = 4 #100
    #should result in an *integer number of steps*:
    new_writeInterval = 1 #0.2

    controlDictupdate(new_start_time, new_end_time, new_writeInterval, case_name)

    #paramters for swimmer movement during simulations
    total_steps = int ((new_end_time - new_start_time) / new_writeInterval)
    time_steps = linspace(0,total_steps,total_steps)
    vel_x = linspace(1,100,total_steps)
    vel_y = linspace(1,100,total_steps)
    rotation = linspace(1,np.pi,total_steps)

    dynamicMeshDictupdate(total_steps, time_steps, vel_x, vel_y, rotation, case_name)

    #do you need to run_mesh also? (changing # of swimmers)
    runSim(case_name, True)