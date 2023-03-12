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
Main.include("linear_param_action.jl")

model = Main.create_model(6, [-15,15,45])
for i in range(20):
    s = np.random.rand(6,1) * (i+1)
    a = random.choice([-15,15,45])
    r = random.random() * (i+1)
    sp = np.random.rand(6,1) * (i+1)
    Main.update_b(model, s, a, r, sp)
    print(model)
for i in range(10):
    print(Main.get_action(model, np.random.rand(6,1) * (i+1), False, True))