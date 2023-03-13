using Random
using LinearAlgebra
"""
Initilize, update and get action from linear model
"""

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇

mutable struct EpsilonGreedyExploration
    ϵ # probability of random arm
end

function (π::EpsilonGreedyExploration)(model, s)
    𝒜, ϵ = model.𝒜, π.ϵ
    if rand() < ϵ
        return rand(𝒜)
    end
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), 𝒜)
end

mutable struct SoftmaxExploration
    λ # precision parameter
    α # precision factor
end

function (π::SoftmaxExploration)(model, s)
    λ, α = π.λ, π.α
    Q(s,a) = lookahead(model, s, a)
    weights = exp.(λ * mean.([Q(s,a) for a in 𝒜]))
    λ *= α
    return rand(Categorical(normalize(weights, 1)))
end

struct GradientQLearning
    𝒜  # action space (assumes 1:nactions)
    γ  # discount
    Q  # parameterized action value function Q(θ,s,a)
    ∇Q # gradient of action value function
    θ  # action value function parameter
    α  # learning rate
end

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.θ, s,a)
end

function update!(model::GradientQLearning, s, a, r, s′)
    𝒜, γ, Q, θ, α = model.𝒜, model.γ, model.Q, model.θ, model.α    
    u = maximum(Q(θ,s′,a′) for a′ in 𝒜) #picks the right θ from create model
    Δ = (r + γ*u - Q(θ,s,a))*model.∇Q(θ,s,a)
    θ[a,:] += α*scale_gradient(Δ, 1)
    return model
end

function create_model(dim_𝒮, num_𝒜)
    """
    Initilizes linear model with zero weights
        @param dim_𝒮: dimension of a singular state
        @param num_𝒜: size of action space
        @return model
    """
    β(s,a) = [s; s.^2]

    Q(θ,s,a) = dot(θ[a,:],β(s,a))
    ∇Q(θ,s,a) = β(s,a)

    # edit size when changing β
    θ = zeros(num_𝒜, 2*dim_𝒮) #initial parameter vector
    α = 0.5 # learning rate
    𝒜 = collect(1:num_𝒜) # number of states
    γ = 0.95 # discount
    model = GradientQLearning(𝒜, γ, Q, ∇Q, θ, α)
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
    model = create_model(6, 3)
    for i in 1:1000
        s = rand(6,1) * 100
        a = rand(1:3)
        r = rand() * 10
        s′ = rand(6,1) * 100
        update!(model, s, a, r, s′)
    end
    print(model.θ)
    for i in 1:10
        println(get_action_sm(model, rand(6,1) * 100, false, true))
    end
end