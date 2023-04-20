#=
        project1.jl -- This is where the magic happens!

    All of your code must either live in this file, or be `include`d here.
=#

#=
    If you want to use packages, please do so up here.
    Note that you may use any packages in the julia standard library
    (i.e. ones that ship with the julia language) as well as Statistics
    (since we use it in the backend already anyway)
=#

# Example:
# using LinearAlgebra

#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")


"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""

function back_tracking_line_search(f, g, x_k, dir, alpha)
    """
    f = objective function, desire MINIMIZE
    g = gradient for f at x_k
    dir = descent direction
    alpha = step size modifier
    percent = percent decrease of alpha each loop
    """
    #percent decrease of alpha until f(x_k+1) <= f(x_k) + step_size
    #where x_k+1 is alpha*dir
    #and step_size = beta*alpha*direction along the gradient
    beta = 1e-4 #shared as a common value for beta on pg 56
    step_size = beta * alpha * (g⋅dir)
    iterator_tracking = 0

    while f(x_k + alpha*dir) >
        iterator_tracking  += 1
        alpha *= percent
    end
    return
end   


function optimize(f, g, x0, n, prob)
    x_best = x0
    return x_best
end