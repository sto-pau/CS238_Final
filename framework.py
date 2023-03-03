'''
simulation look for reinforcement learning
note the Qlearning code called
contains the model including exploration policy
'''
import numpy as np
from numpy import linspace
import subprocess
import os

def controlDictupdate(new_start_time, new_end_time, new_writeInterval, case_name):
    '''
    new_start_time = time to start simulation
    new_end_time = time to end simulation
    new_writeInterval = interval length, linear interpolation in betwee,
    one folder will be output for each of the intervals
    case_name = simulation folder
    '''

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
    '''
    total_steps = total number of time instances to set swimmer state
    time_steps = time in simulation that corresponds to each of the instances
    vel_x, vel_y, rotation = amount change for each of the state
    NOTE: velocity will actually depend on time between intervals
    case_name = simulation folder
    '''

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

    #filepath to simulation folder
    case_name = 'CS238_dummy_case'

    ''''''''''''''''''''''setup section'''''''''''''''''''''''''''''''''''''
    run model to reach steady state
    10 seconds without moving the swimmer
    '''
    init_start_time = 0
    init_end_time  = 10
    init_writeInterval = 10 
    controlDictupdate(init_start_time, init_end_time, init_writeInterval, case_name)    
    
    init_total_steps = 2 #start and end only
    init_time_steps = np.array([0, end_time])
    #initial action is no motion
    vel_x = np.array([0, 0])
    vel_y = np.array([0, 0])
    rotation = np.array([0, 0])    
    dynamicMeshDictupdate(init_total_steps, init_time_steps, vel_x, vel_y, rotation, case_name)
    
    '''to be updated, run first simulation, get initial state and reward

    runSim(case_name, True)
    state_prime, rewards = ReadOutput(start_time, sim_step_length+start_time)
    state = state_prime #for first case, no motion, no change in state
    '''

    ''''''''''''''''''''''learning loop'''''''''''''''''''''''''''''''''''''    
    start time is where setup section ended
    '''
    #evaluation loop
    start_t = end_time #start at the end of init period
    duration  = 10 #total learning length
    sim_step_length = 0.1 
    #from 10.1 to 20 in steps of 0.1
    time_steps = linspace(end_time+sim_step_length, end_time+duration, int(duration/sim_step_length))
    for end_t in time_steps:  

        '''to be updated call Q_learning for action
        action = Q_learning.getPolicy(state)
        '''

        #set action states, linear interpolation in between        
        rotation[1] = action #set end action for next simulation based on Q
        controlDictupdate(start_t, end_t, sim_step_length, case_name)
        sim_time_steps = np.array([start_t, end_t])
        dynamicMeshDictupdate(2, sim_time_steps, vel_x, vel_y, rotation, case_name)  
        '''to be updated simulate to get next state and reward 

        runSim(case_name, True)
        state_prime, reward = ReadOutput(start_time, sim_step_length+start_time)
        '''

        '''to be updated, update model

        Q_learning.Update(state, action, reward, state_prime)'''

        '''to be updated save values for next loop
        
        state = state_prime'''        
        start_t = end_t #set start time for next loop
        rotation[0] = rotation[1] #set start action as end of last state

    ''''''''''''''''''''''evaluation section'''''''''''''''''''''''''''''''''
    #policy = Q_learning.getPolicy()
    
    #evaluation loop
    eval_start = end_t #start at the end of init period
    eval_duration  = 5 #total learning length
    eval_step_length = 0.1 
    #from 20.1 to 25 in steps of 0.1
    eval_steps = linspace(end_t+eval_step_length, end_t+eval_duration, int(eval_duration/eval_step_length))

    actions = []
    rewards = []

    for eval_end in eval_steps:  

        '''to be updated call Q_learning for action
        action = Q_learning.getPolicy(state)
        actions.extend(action)
        '''

        #set action states, linear interpolation in between        
        rotation[1] = action #set end action for next simulation based on Q
        controlDictupdate(start_t, end_t, sim_step_length, case_name)
        sim_time_steps = np.array([start_t, end_t])
        dynamicMeshDictupdate(2, sim_time_steps, vel_x, vel_y, rotation, case_name)  
        '''to be updated simulate to get next state and reward 

        runSim(case_name, True)
        state_prime, reward = ReadOutput(start_time, sim_step_length+start_time)
        rewards.extend(reward)
        '''

        '''to be updated save values for next loop
        
        state = state_prime'''        
        start_t = end_t #set start time for next loop
        rotation[0] = rotation[1] #set start action as end of last state

   #evaluation actions + rewards