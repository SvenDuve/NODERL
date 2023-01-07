
@with_kw mutable struct Parameter
    environment::String = "MountainCarContinuous-v0"
    state_size::Int = 2
    action_size::Int = 1
    action_bound::Float64 = 1.0
    action_bound_high::Array = [1.0]
    action_bound_low::Array = [-1.0]
    batch_size::Int = 128
    DDPG_batch::Int = 128
    batch_length::Int = 40
    mem_size::Int = 1000000
    frames::Int = 0
    env_steps::Int = 0
    train_start::Int = 1000 
    max_episodes::Int = 2000
    max_episodes_length::Int = 1000
    max_episodes_length_mb::Int = 1000
    Sequences::Int = 10
    trainloops_mb::Int = 10
    critic_hidden::Array = [(200, 200)]
    actor_hidden::Array = [(200, 200)]
    reward_hidden::Array = [(200, 200)]
    dynode_hidden::Array = [(200, 200)]
    γ::Float64 = 0.99
    noise_type::String = "gaussian"
    τ_actor::Float64 = 0.1
    τ_critic::Float64 = 0.5
    η_actor::Float64 = 0.0001 #lr
    η_critic::Float64 = 0.01 #lr
    η_node::Float64 = 0.001 #lr
    η_reward::Float64 = 0.001 #lr
    H::Int = 200
    m::Int = 1000
    dT::Float32 = 0.01
    model_loss::Array = []
    reward_loss::Array = []
    total_rewards::Array = []
    world_rewards::Array = []
end




function resetParameters(p)
    
    newP = Parameter(p; state_size=env.observation_space.shape[1],
        action_size=env.action_space.shape[1],
        action_bound=env.action_space.high[1],
        action_bound_high=env.action_space.high,
        action_bound_low=env.action_space.low)
    return newP
end



function 𝒩(ou::OrnsteinUhlenbeck)
    dx = ou.θ .* (ou.μ .- ou.X)
    dx = dx .+ ou.σ .* randn(Float32, length(ou.X))
    ou.X = ou.X .+ dx
end



function 𝒩(gn::GaussianNoise)
    rand(Normal(gn.μ, gn.σ))
end


function setNoise(p::Parameter) 
    if p.noise_type == "gaussian"
        global noise = GaussianNoise(0.0f0, 0.1f0)
    else
        global noise = OrnsteinUhlenbeck(0.0f0, 0.15f0, 0.5f0, [0.0f0])
    end
end

# function setNoise() end


mutable struct Episode
    env::PyObject
    l::Learner
    p::Parameter
    total_reward::Float64 # total reward of the episode
    last_reward::Float64
    niter::Int     # current step in this episode
    freq::Int       # number of steps between choosing actions
    maxn::Int       # max steps in an episode - should be constant during an episode
    episode::Array

    function Episode(env::PyObject, l::Learner, p::Parameter)

        total_reward, last_reward = 0.0, 0.0
        niter = 1
        freq = 1
        maxn = p.max_episodes_length
        episode = []
        new(env, l, p, total_reward, last_reward, niter, freq, maxn, episode)
    end
end



function (e::Episode)()

    s::Vector{Float32} = e.env.reset()
    r::Float64 = 0.0
    a::Vector{Float64} = [0.0] # check action space
    t::Bool = false

    for i in 1:e.maxn
        a = action(e.l.action_type, e.l.train, s, e.p)
        s′, r, t, _ = e.env.step(a)
        #@show a, s′, r, t
        e.total_reward += r
        #@show e.total_reward
        append!(e.episode, [(s, a, r, s′, t)])
        s = s′
        if t
            return e
            #s = env.reset()
        end
    end

    return e

end



function action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    vcat(clamp.(μϕ(s) .+ vcat([𝒩(noise) for i in 1:p.action_size]...) * m, -p.action_bound, p.action_bound)...)
end


function action(t::ActionSelection, m::Bool, s::Vector{Float32}, p::Parameter)
    return Vector{Float32}(env.action_space.sample())
end

function action(t::MPC, m::Bool, s::Vector{Float32}, p::Parameter) 

    Sequences = []

    for k in 1:2  
        append!(Sequences, [[[rand(Uniform(el[1], el[2])) for el in zip(p.action_bound_low, p.action_bound_high)] for j in 1:2]])
    end

    R = []
    S = []
    r = []
    s_ = []

    for Sequence in Sequences
        for a in Sequence
            append!(r, Rϕ(vcat(s, a)) |> first)
            s = fθ(vcat(s, a))
            #s = fθ(vcat(s, a))
            append!(s_, s)
        end
        append!(R, sum(r))
        append!(S, [s_])
        r = []
        s_ = []
    end

    return Vector{Float32}(Sequences[argmax(R)][1])

end



function greetings()
    println("Heeeeelllooo RL")
end

function getVals(algorithm::T) where T <: Union{DDPG, SAC, TW3} 
    return p, μϕ
end

function getVals(algorithm::T) where T <: Union{DynaWorldModel, NODEModel}
    return p, fθ, Rϕ
end


macro getName(x)
    quote
        $(string(x))
    end
end
