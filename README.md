$ python3 -m pip install --user julia  

$ python3  
>>> import julia  
>>> julia.install()  

julia> using Pkg  
julia> Pkg.add("PyCall")  
julia> Pkg.add("Distributions")  
  
Example in pyjul.py  
  
Use a copy test_framework for submitting new runs