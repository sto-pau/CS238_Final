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

model = Main.create_model(6, 3)
for i in range(200):
    s = np.random.rand(6,1)
    a = random.randint(1,3)
    norm = np.linalg.norm(s)
    if s[0] < 0.5 and a == 1: #stay close to center
        r = 20
    elif s[0] > 0.5 and a == 2: #better to stay in the center
        r = random.random() * 20 #* (i+1)
    else: #a = don't reward any big motion regardless of state
        r = 0 
    sp = np.random.rand(6,1)
    print("norm a r:", norm, a, r)
    Main.update_b(model, s, a, r, sp)
    #print(model)
for i in range(10):
    s = np.random.rand(6,1)
    print(s)
    print(Main.get_action_sm(model, s, False, True))