# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:37:45 2023

@author: kozak
"""

## ------------------------------------------------------------------------  ##
"Load Packages"
import pyvista as pv
import numpy as np
import json

## ------------------------------------------------------------------------  ##
"Setup"
# case name
case_name = 'CS238_dummy_case'
# start time of the simulation: backGround/system/controlDict
    # the start time names the postprocessing folder name 
time_start = 0 
# times of interset to process
time_analysis = np.array([0.2,0.4,0.6])

## ------------------------------------------------------------------------  ##
"Read Rewards: Coefficeints"
# rewards
    # coefficients
    # Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f) Cs(r)             
    # 0    1  2     3     4  5     6     7       8      9     10 11    12
path_rewards = case_name+'/backGround/postProcessing/forceCoeffs_object/' + str(time_start) + '/'
file_rewards = 'coefficient.dat'
data_rewards = np.loadtxt(path_rewards+file_rewards,skiprows=13)
rewards = np.zeros((len(time_analysis),13))
for t,time in enumerate(time_analysis):
    rewards[t] = data_rewards[data_rewards[:,0]==time,:]
    
## ------------------------------------------------------------------------  ##
"Read States: Forces/Moments and Pressure"
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
    
        
        