using Random
using LinearAlgebra
using Statistics
using Distributions
"""
Initilize, update and get action from linear model
"""

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇

mutable struct EpsilonGreedyExploration
    ϵ # probability of random arm
    α # decay constant between 0-1
end

function (π::EpsilonGreedyExploration)(model, s)
    𝒜, ϵ, α = model.𝒜, π.ϵ, model.α
    if rand() < ϵ
        return rand(𝒜)
    end
    ϵ *= α
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), 𝒜)
end

mutable struct SoftmaxExploration
    λ # precision parameter, choose small value so it doesn't go to infinity
    α # precision factor
end

function (π::SoftmaxExploration)(model, s)
    𝒜, λ, α = model.𝒜, π.λ, π.α
    Q(s,a) = lookahead(model, s, a)
    weights = exp.(λ * ([Q(s,a) for a in 𝒜]))
    λ *= α
    weights = normalize(weights, 1)
    p = (1-sum(A -> map(x -> isnan(x) ? zero(x) : x, A), weights)) / count(i->(isnan(i)), weights)
    replace!(weights, NaN=>p)
    return rand(Categorical(weights))
end

mutable struct UCB1Exploration
    c # exploration constant
end

function bonus(counts, a)
	N = sum(counts)
	Na = counts[a]
    return sqrt(log(N)/Na)
end

function (π::UCB1Exploration)(model, s)
    Q(s,a) = lookahead(model, s, a)
    ρ = [Q(s,a) for a in model.𝒜]
    u = ρ .+ π.c*[bonus(model.N, a) for a in model.𝒜]
    return argmax(u)
end

"""
DOES NOTE WORK AS IT IS RIGHT NOW
since Q(s,a) is a scalar, not a vector or distribution 

mutable struct QuantileExploration
    α # quantile (e.g., 0.95)
end

function (π::QuantileExploration)(model, s)
    Q(s,a) = lookahead(model, s, a)
    return argmax(quantile(a->Q(s,a), π.α), 𝒜)
end
"""

struct GradientQLearning
    𝒜  # action space 
    γ  # discount
    Q  # parameterized action value function Q(θ,s,a)
    ∇Q # gradient of action value function
    θ  # action value function parameter
    α  # learning rate
    N  # number of times action was taken + pseudocounts
end

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.θ, s,a)
end

function update!(model::GradientQLearning, s, a, r, s′)
    𝒜, γ, Q, θ, α, N = model.𝒜, model.γ, model.Q, model.θ, model.α, model.N   
    u = maximum(Q(θ,s′,a′) for a′ in 𝒜) #picks the right θ from create model
    Δ = (r + γ*u - Q(θ,s,a))*model.∇Q(θ,s,a)
    θ[a,:] += α*scale_gradient(Δ, 1)
    N[a] += 1
    return model
end

function β_helper(s,a,𝒜)
    a = Dict(collect(1:length(𝒜)).=> 𝒜)[a]
    return [s;s.^3;[1/a]]
end

function create_model(dim_𝒮,𝒜)
    """
    Initilizes linear model with zero weights
        @param dim_𝒮: dimension of a singular state
        @param 𝒜: action space
        @return model
    """
    # refer to line 115 in framework.py
    β(s,a) = β_helper(s,a,𝒜)

    Q(θ,s,a) = dot(θ[a,:],β(s,a))
    ∇Q(θ,s,a) = β(s,a)

    # edit size when changing β
    θ = zeros(length(𝒜), 2*dim_𝒮+1) #initial parameter vector
    α = 0.5 # learning rate
    𝒜′ = collect(1:length(𝒜)) # number of states
    γ = 0.95 # discount
    N = ones(length(𝒜)) # pseudocounts for UCB1
    model = GradientQLearning(𝒜′, γ, Q, ∇Q, θ, α, N)
    return model
end

function get_action(model::GradientQLearning, π, s, rand, test)
    """
    Returns action for current state s
        @param model: initilized linear model
        @param π: exploration policy
        @param s: current state
        @param rand: explore only (FULLY random) << initializing phase >>
        @param test: exploit only << testing phase >>
        @return action
    """
    if rand
        return rand(model.𝒜)
    elseif test
        Q(s,a) = lookahead(model, s, a)
        return argmax(a->Q(s,a), model.𝒜)
    end
    return π(model, s)
end

function example_run()
    """
    Demos how to use
    """
    model = create_model(18,[60,30,45])
    exploration_policy = UCB1Exploration(1)
    for i in 1:1000
        s = rand(18,1) * 100
        a = rand(1:3)
        r = rand() * 10
        s′ = rand(18,1) * 100
        update!(model, s, a, r, s′)
    end
    # print(model.θ)
    for i in 1:10
        println(get_action(model, exploration_policy, rand(18,1) * 100, false, false))
    end
end