'''
simulation look for reinforcement learning
note the Qlearning code called
contains the model including exploration policy
'''
import numpy as np
import subprocess
import os
import json
import pyvista as pv

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.include("linear_model.jl")


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

def runIntialSim(case_name):
    org_path = os.getcwd()
    os.chdir(org_path + '/' + case_name)
    subprocess.run(['./run_mesh.sh'])
    subprocess.run(['./run_solver_i.sh'])
    os.chdir(org_path)

def runSim(case_name):
    org_path = os.getcwd()
    os.chdir(org_path + '/' + case_name)
    subprocess.run(['./run_solver.sh'])
    os.chdir(org_path)


def get_Rewards_States(case_name,time_start,fms_flag,time_analysis):
    '''
    --- --- input
    case_name = simulation folder
    time_start = what is the time_start in the control dict
    fms_flag = use forces and moments to get the state (else it is the pressure)
    time_analysis = array with what times we want to analyze
    --- --- output
    state_prime, reward

    #
    '''
    if time_start == round(time_start):
        time_start = round(time_start)

    path_rewards = case_name+'/backGround/postProcessing/forceCoeffs_object/' + str(time_start) + '/'
    file_rewards = 'coefficient.dat'
    data_rewards = np.loadtxt(path_rewards+file_rewards,skiprows=13)
    rewards = np.zeros((len(time_analysis),13))
    for t,time in enumerate(time_analysis):
	# rewards
    		# coefficients
    		# Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f) Cs(r)
    		# 0    1  2     3     4  5     6     7       8      9     10 11    12
        rewards[t] = data_rewards[data_rewards[:,0]==time,:]

    if fms_flag: #force,moment state
        # states:
            # forces and moments
            # time total_x total_y total_z	pressure_x pressure_y pressure_z  viscous_x viscous_y viscous_z
            # 0    1       2       3        4          5          6            7        8         9
        path_states_forces = case_name+'/backGround/postProcessing/forces_object/' + str(time_start) + '/'
        file_states_forces = 'force.dat'
        data_states_forces = np.loadtxt(path_states_forces+file_states_forces,skiprows=4)
        file_states_moments = 'moment.dat'
        data_states_moments = np.loadtxt(path_states_forces+file_states_moments,skiprows=4)
        states_forces = np.zeros((len(time_analysis),10))
        states_moments = np.zeros((len(time_analysis),10))
        for t,time in enumerate(time_analysis):
            states_forces[t] = data_states_forces[data_states_forces[:,0]==time,:]
            states_moments[t] = data_states_moments[data_states_moments[:,0]==time,:]
        return np.hstack((states_forces,states_moments)), rewards

    else:
            # pressure along airfoil
        path_states_p_series = case_name+'/backGround/VTK/'
        file_states_p_series = 'backGround.vtm.series'
            # Open the JSON file and load the data
        with open(path_states_p_series+file_states_p_series, 'r') as f:
            data_states_forces = json.load(f)
        data_states_forces = data_states_forces['files']
        states_p_filenames = [""] * len(time_analysis)
            # loop through and find relevant times
        for file_data in data_states_forces:
            name = file_data['name']
            time = file_data['time']
            if time in time_analysis:
                states_p_filenames[np.where(time_analysis==time)[0][0]] = name.split('.')[0]
        states_p = np.zeros((len(time_analysis),652))
        for p,file_states_p in enumerate(states_p_filenames):
            path_states_p = case_name+'/backGround/VTK/'+file_states_p+'/boundary/wing.vtp'
            reader = pv.get_reader(path_states_p)
            reader.disable_all_cell_arrays()
            reader.disable_all_point_arrays()
            reader.enable_point_array('p')
            states_p[p,:] = np.array(reader.read().point_data['p'])
        return states_p, rewards

def get_Rewards_States_list(case_name,sim_step_length,fms_flag,time_analysis):
    '''
    --- --- input
    case_name = simulation folder
    sim_step_length = time step size in simulation
    fms_flag = use forces and moments to get the state (else it is the pressure)
    time_analysis = array with what times we want to analyze
    --- --- output
    state_prime, reward

    #
    '''
    rewards = np.zeros((len(time_analysis),13))
    states_forces = np.zeros((len(time_analysis),10))
    states_moments = np.zeros((len(time_analysis),10))
    states_p = np.zeros((len(time_analysis),652))

    if ~fms_flag: #force,moment state
        path_states_p_series = case_name+'/backGround/VTK/'
        file_states_p_series = 'backGround.vtm.series'
            # Open the JSON file and load the data
        with open(path_states_p_series+file_states_p_series, 'r') as f:
            data_states_forces = json.load(f)
        data_states_forces = data_states_forces['files']
        states_p_filenames = [""] * len(time_analysis)
            # loop through and find relevant times
        for file_data in data_states_forces:
            name = file_data['name']
            time = file_data['time']
            if time in time_analysis:
                states_p_filenames[np.where(time_analysis==time)[0][0]] = name.split('.')[0]

    for t,time_start in enumerate(time_analysis):
        if np.round(time_start-sim_step_length,6) == round(time_start-sim_step_length):
            tt = round(time_start-sim_step_length)
        else:
            tt = np.round(time_start-sim_step_length,6)

        path_rewards = case_name+'/backGround/postProcessing/forceCoeffs_object/' + str(tt) + '/'
        file_rewards = 'coefficient.dat'
        data_rewards = np.loadtxt(path_rewards+file_rewards,skiprows=13)
        rewards[t] = data_rewards[data_rewards[:,0]==time_start,:]

        if fms_flag: #force,moment state
            # states:
                # forces and moments
                # time total_x total_y total_z	pressure_x pressure_y pressure_z  viscous_x viscous_y viscous_z
                # 0    1       2       3        4          5          6            7        8         9
            path_states_forces = case_name+'/backGround/postProcessing/forces_object/' + str(tt) + '/'
            file_states_forces = 'force.dat'
            data_states_forces = np.loadtxt(path_states_forces+file_states_forces,skiprows=4)
            file_states_moments = 'moment.dat'
            data_states_moments = np.loadtxt(path_states_forces+file_states_moments,skiprows=4)
            states_forces[t] = data_states_forces[data_states_forces[:,0]==time_start,:]
            states_moments[t] = data_states_moments[data_states_moments[:,0]==time_start,:]

        else:
            # pressure along airfoil
            file_states_p = states_p_filenames[t]
            path_states_p = case_name+'/backGround/VTK/'+file_states_p+'/boundary/wing.vtp'
            reader = pv.get_reader(path_states_p)
            reader.disable_all_cell_arrays()
            reader.disable_all_point_arrays()
            reader.enable_point_array('p')
            states_p[t,:] = np.array(reader.read().point_data['p'])

    if fms_flag:
        return np.hstack((states_forces,states_moments)), rewards
    return states_p, rewards




if __name__ == '__main__':

    #filepath to simulation folder
    case_name = 'test_policy_epsilon0.3' #123

    # use states that are defined by forces and moments or pressure
    fms_flag = True
    ''''''''''''''''''''''setup section'''''''''''''''''''''''''''''''''''''
    run model to reach steady state
    10 seconds without moving the swimmer
    '''
    init_start_time = 0
    init_end_time  = 0.1
    init_writeInterval = init_end_time-init_start_time
    controlDictupdate(init_start_time, init_end_time, init_writeInterval, case_name)

    init_total_steps = 2 #start and end only
    init_time_steps = np.array([0, init_end_time+1e-6])
    #initial action is no motion
    vel_x = np.array([0, 0])
    vel_y = np.array([0, 0])
    rotation = np.array([0, 0])
    dynamicMeshDictupdate(init_total_steps, init_time_steps, vel_x, vel_y, rotation, case_name)
    # update control dict
    eol = os.linesep
    path = case_name + '/backGround/system/controlDict'
    with open(path, 'r') as file:
        file_lines = file.readlines()
    file.close()
    for line_number , line_content in enumerate(file_lines):
        if line_content.startswith("deltaT"):
            file_lines[line_number] = "deltaT          {};{}".format(1e-5, eol) #10 spaces
    with open(path, 'w') as file:
        file.writelines(file_lines)
    file.close()
    # run first simulation
    runIntialSim(case_name)
    # get the intial state
    state_prime, _ = get_Rewards_States(case_name,init_start_time,fms_flag,[init_end_time])
    state = state_prime #for first case, no motion, no change in state
    # update control dict
    eol = os.linesep
    path = case_name + '/backGround/system/controlDict'
    with open(path, 'r') as file:
        file_lines = file.readlines()
    file.close()
    for line_number , line_content in enumerate(file_lines):
        if line_content.startswith("deltaT"):
            file_lines[line_number] = "deltaT          {};{}".format(1e-3, eol) #10 spaces
    with open(path, 'w') as file:
        file.writelines(file_lines)
    file.close()

    ''''''''''''''''''''''learning loop'''''''''''''''''''''''''''''''''''''
    start time is where setup section ended
    '''
    #learning loop
    start_t = init_end_time #start at the end of init period
    duration  = 0.4 #total learning length
    sim_step_length = 0.2
    #from 10.1 to 20 in steps of 0.1
    # round is required because of numerical precison issues of linspace
    time_steps = np.round(np.linspace(start_t+sim_step_length, start_t+duration, int(duration/sim_step_length)),6)

    '''model setup'''
    state_dim = len(state_prime.flatten()) # linear model expects 1D input
    action_space = list(np.arange(-15,15,2.5)) # change as needed
    num_actions = len(action_space)
    model = Main.create_model(state_dim, num_actions)
    exploration_policy = Main.EpsilonGreedyExploration(0.3) # change exploration policy

    list_of_actions_learning = []
    list_of_states_learning = []
    list_of_rewards_learning = []

    for end_t in time_steps:
        list_of_states_learning.append(state)
        '''call linear model to get action index [1,12]'''
        '''NOTE THAT REWARD MUST BE A SINGLE NUMBER'''
        action = Main.get_action(model, exploration_policy, state.flatten(), False, False)
        list_of_actions_learning.append(action_space[action-1])
        rotation[1] += action_space[action-1] #set end action for next simulation based on Q
        rotation[1] = rotation[1] % 360 if rotation[1] > 0 else rotation[1] % -360
        controlDictupdate(start_t, end_t, sim_step_length, case_name)
        sim_time_steps = np.array([start_t, end_t+1e-6])
        dynamicMeshDictupdate(2, sim_time_steps, vel_x, vel_y, rotation, case_name)
        #simulate to get next state and reward
        runSim(case_name)
        #get the state and rewards
        state_prime, reward = get_Rewards_States(case_name,start_t,fms_flag,[end_t])
        list_of_rewards_learning.append(reward[0][:])
        '''update model'''
        Main.update_b(model, state.flatten(), action, reward[0][1], state_prime.flatten())
        #updated save values for next loop
        state = state_prime
        start_t = end_t #set start time for next loop
        rotation[0] = rotation[1] #set start action as end of last state

    ''''''''''''''''''''''evaluation section'''''''''''''''''''''''''''''''''
    # return to zero --> take 5 seconds
    r_0_t = 8 # return to zero time
    controlDictupdate(end_t,end_t+r_0_t,r_0_t,case_name)
    sim_time_steps = np.array([end_t,end_t+r_0_t*0.7, end_t+r_0_t+1e-6])
    dynamicMeshDictupdate(3, sim_time_steps, np.array([0,0,0]), np.array([0,0,0]), np.array([rotation[0],0,0]), case_name)
    runSim(case_name)
    state_prime, _ = get_Rewards_States(case_name,end_t,fms_flag,[end_t+r_0_t])
    state = state_prime #for first case, no motion, no change in state
    rotation[0] = 0
    #evaluation loop
    eval_start = end_t+r_0_t #start at the of training + 3 for return to zero
    eval_duration  = 1 #total learning length
    eval_step_length = 0.1
    #from 20.1 to 25 in steps of 0.1
    eval_steps = np.round(np.linspace(eval_start+eval_step_length, eval_start+eval_duration, int(eval_duration/eval_step_length)),6)
    list_of_actions = []
    list_of_states = []
    for eval_end in eval_steps:
        list_of_states.append(state)
        '''call linear model to get action index [1,12]'''
        action = Main.get_action(model, exploration_policy, state.flatten(), False, True)
        list_of_actions.append(action_space[action-1])
        rotation[1] += action_space[action-1] #set end action for next simulation based on Q
        rotation[1] = rotation[1] % 360 if rotation[1] > 0 else rotation[1] % -360
        controlDictupdate(eval_start, eval_end, eval_step_length, case_name)
        sim_time_steps = np.array([eval_start, eval_end+1e-6])
        dynamicMeshDictupdate(2, sim_time_steps, vel_x, vel_y, rotation, case_name)
        #simulate to get next state and reward
        runSim(case_name)
        #get the state and rewards
        state, _ = get_Rewards_States(case_name,eval_start,fms_flag,[eval_end])
        #updated save values for next loop
        state = state_prime
        eval_start = eval_end #set start time for next loop
        rotation[0] = rotation[1] #set start action as end of last state

    #evaluation actions + rewards
    _, reward = get_Rewards_States_list(case_name,eval_step_length,fms_flag,eval_steps)
    total_reward = 0
    # write
    with open(case_name+'/actions.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in list_of_actions) )
        fp.write('\n')
    with open(case_name+'/states.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in list_of_states) )
        fp.write('\n')
    with open(case_name+'/rewards.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in reward) )
        fp.write('\n')
    with open(case_name+'/rewards_CD.txt', 'w') as fp:
        fp.write('\n'.join( str(a[1]) for a in reward) )
        fp.write('\n')

    py_obj_str = repr(model)
    with open(case_name+'/model.txt', 'w') as f:
       f.write(py_obj_str)

    with open(case_name+'/actions_L.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in list_of_actions_learning) )
        fp.write('\n')
    with open(case_name+'/states_L.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in list_of_states_learning) )
        fp.write('\n')
    with open(case_name+'/rewards_L.txt', 'w') as fp:
        fp.write('\n'.join( str(a) for a in list_of_rewards_learning) )
        fp.write('\n')
    with open(case_name+'/rewards_CD_L.txt', 'w') as fp:
        fp.write('\n'.join( str(a[1]) for a in list_of_rewards_learning) )
        fp.write('\n')

    print('eval_steps: ', eval_steps)
    for i in range(len(eval_steps)):
      total_reward += reward[i][1]
    print('list of actions: ', list_of_actions)
    print('total reward:  ', total_reward)
