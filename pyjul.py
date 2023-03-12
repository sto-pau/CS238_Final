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
for i in range(2000):
    s = np.random.rand(6,1) * (i+1)
    a = random.randint(1,3)
    r = random.random() * 100 #* (i+1)
    sp = np.random.rand(6,1) * (i+1)
    print("a r:", a, r)
    Main.update_b(model, s, a, r, sp)
    #print(model)
for i in range(10):
    state = np.random.rand(6,1) * (i+1)
    #print(state)
    print(Main.get_action_sm(model, state, False, True))