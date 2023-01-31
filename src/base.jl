
@with_kw mutable struct Parameter
    environment::String = "MountainCarContinuous-v0"
    state_size::Int = 2
    state_high::Array = [0.6, 0.07]
    state_low::Array = [-1.2, -0.07]
    action_size::Int = 1
    action_bound::Float64 = 1.0
    action_bound_high::Array = [1.0]
    action_bound_low::Array = [-1.0]
    batch_size::Int = 128 # general batch size
    batch_size_episodic::Int = 1 # Batch sixe for the episodic model
    DDPG_batch::Int = 128 # Batch size for the DDPG Agent
    batch_length::Int = 40 # episode length per sample for the model
    mem_size::Int = 1000000 # Buffer size
    env_steps::Int = 0 
    frames::Int = 0
    train_start::Int = 1000 # Exploration steps without learning
    max_episodes::Int = 2000
    max_episodes_length::Int = 1000
    max_episodes_length_mb::Int = 1000
    episode_length::Array = [] # Captures episode length for every interaction
    world_episode_length::Array = [] #Captures the episode length while agent training in world
    Sequences::Int = 10 # # number of episodes to run for DynaWorldModel
    model_episode_length::Int = 400 # Number of steps to simulate 
    model_episode_retrain::Int = 50 # Number of episodes to retrain the model
    trainloops_mb::Int = 10
    train_fr::Int = 5 # how often to train DDPG, in MB
    critic_hidden::Array = [(200, 200)]
    actor_hidden::Array = [(200, 200)]
    reward_hidden::Array = [(200, 200)]
    dynode_hidden::Array = [(200, 200)]
    γ::Float64 = 0.99
    noise_type::String = "gaussian" # action noise type, either "gaussian" or "ou"
    gaussian_μ::Float64 = 0.0
    gaussian_σ::Float64 = 0.1
    ou_μ::Float64=0.0
    ou_θ::Float64=0.15
    ou_σ::Float64=0.2
    τ_actor::Float64 = 0.1 # base/ target weigthing
    τ_critic::Float64 = 0.5
    η_actor::Float64 = 0.0001 #lr for the actor
    η_critic::Float64 = 0.01 #lr for the critic
    η_node::Float64 = 0.001 #lr for the NODE
    η_reward::Float64 = 0.001 #lr for the reward
    H::Int = 200
    m::Int = 1000
    dT::Float32 = 0.01
    model_loss::Array = []
    reward_loss::Array = []
    validation_loss::Array = []
    total_rewards::Array = []
    world_rewards::Array = []
end




function resetParameters(p)
    
    newP = Parameter(p; state_size=env.observation_space.shape[1],
        state_high = env.observation_space.high,
        state_low = env.observation_space.low,
        action_size=env.action_space.shape[1],
        action_bound=env.action_space.high[1],
        action_bound_high=env.action_space.high,
        action_bound_low=env.action_space.low)
    return newP
end



function 𝒩(ou::OrnsteinUhlenbeck)
    dx = ou.θ .* (ou.μ .- ou.X)
    #@show dx
    dx = dx .+ ou.σ .* randn(Float32, length(ou.X))
    #@show dx
    ou.X = ou.X .+ dx
    return ou.X
end



function 𝒩(gn::GaussianNoise)
    rand(Normal(gn.μ, gn.σ))
end

function 𝒩(nl::NoiseFree) return false end


function setNoise(p::Parameter) 
    if p.noise_type == "gaussian"
        global noise = GaussianNoise(p.gaussian_μ, p.gaussian_σ)
    elseif p.noise_type == "ou"
        global noise = OrnsteinUhlenbeck(p.ou_μ, p.ou_θ, p.ou_σ, zeros(p.action_size))
    else
        global noise = NoiseFree()
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
    a::Vector{Float32} = [0.0] # check action space
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
            append!(e.p.episode_length, i) 
            e.env.close()
            return e
            #s = env.reset()
        end
    end

    e.env.close()

    return e

end



function getModelEpisode(env, l, p)
    ep = []

    s = env.reset()

    for i in 1:p.model_episode_length

        # a = Vector{Float32}([rand(Uniform(el[1], el[2])) for el in zip(p.action_bound_low, p.action_bound_high)]) 
        a = Vector{Float32}(action(l.action_type, l.train, s, p))
        s′ = Vector{Float32}(s .+ p.dT .* fθ(vcat(s, a)))
        r = Rϕ(vcat(s, a)) |> first |> Float64
        # randS = [rand(Uniform(el[1], el[2])) for el in zip(p.state_low, p.state_high)]
        # randA = [rand(Uniform(el[1], el[2])) for el in zip(p.action_bound_low, p.action_bound_high)]
        append!(ep, [(s, a, r, s′, false)])
        s = s′

    end

    return ep

end



function action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    s = reshape(s, (p.state_size, 1))
    #@show noise.X
    #@show vcat([𝒩(noise) for i in 1:p.action_size]...)
#    @show vcat(clamp.(μϕ(s) .+ vcat([𝒩(noise) for i in 1:p.action_size]...) * m, -p.action_bound, p.action_bound)...)

    # return vcat(clamp.(μϕ(s) .+ vcat([𝒩(noise) for i in 1:p.action_size]...) * m, -p.action_bound, p.action_bound)...)
    return vcat(clamp.(μϕ(s) .+ vcat(𝒩(noise)...) * m, -p.action_bound, p.action_bound)...)
end

function action(t::ActionSelection, m::Bool, s::Vector{Float32}, p::Parameter)
    return env.action_space.sample()
end

function action(t::Randomized, m::Bool, s::Vector{Float32}, p::Parameter)
    return env.action_space.sample() .+ vcat(𝒩(noise)...) * m
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

    return Sequences[argmax(R)][1]

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
