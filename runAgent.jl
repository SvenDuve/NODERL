using UnicodePlots
using NODERL
using Hyperopt 


macro getName(x)
    quote
        $(string(x))
    end
end

# set objects to iterate over here:



function objective(τ_actor, τ_critic, η_actor, η_critic) 

    p, μϕ = trainLearner(Learner(DDPG(),
    Online(),
    Clamped()),
    Parameter(environment="MountainCarContinuous-v0",
    train_start = 10000,
    max_episodes = 100,
    noise_type = "none",
    η_actor = η_actor,
    η_critic = η_critic,
    τ_actor= τ_actor,
    τ_critic= τ_critic));

    # file = "lract" * string(η_actor) * "lrcr" * string(η_critic) * "taua" * string(τ_actor) * "tcr" = string(τ_critic)

    replPlots(DDPG(), "file", p)

    return -sum(p.total_rewards[end-2, end])


end

                               
ho = @hyperopt for i=20,
    sampler = RandomSampler(),     
    #episodes = StepRange(10, 1, 11),
    #batchsize = StepRange(32, 2, 36),
    τ_actor = LinRange(0.01, 0.2, 10), 
    τ_critic = LinRange(0.01, 0.2, 10), 
    η_actor = exp10.(LinRange(-4, -3, 10)),
    η_critic = exp10.(LinRange(-4, -3, 10))
# delta = StepRange(10,5, 25),
# lr =  exp10.(LinRange(-4,-3,10)),
# mm =  LinRange(0.75,0.95,5),
# day0 = StepRange(5,3, 10)
    @show objective(τ_actor, τ_critic, η_actor, η_critic)
end

best_params, min_f = ho.minimizer, ho.minimum

println("Best Parameters " * string(best_params))










# DDPG_actor01_critic0025 = (Learner(DDPG(), 
# Online(), 
# Clamped()), 
# Parameter(environment="MountainCarContinuous-v0",
# train_start = 10000,
# max_episodes = 50,
# max_episodes_length = 1000,
# batch_size=128,
# η_actor = 0.001,
# η_critic = 0.001,
# τ_actor=0.1,
# τ_critic=0.025));

# DDPG_actor005_critic00125 = (Learner(DDPG(), 
# Online(), 
# Clamped()), 
# Parameter(environment="MountainCarContinuous-v0",
# train_start = 10000,
# max_episodes = 50,
# max_episodes_length = 1000,
# batch_size=128,
# η_actor = 0.001,
# η_critic = 0.001,
# τ_actor=0.05,
# τ_critic=0.0125));




# p, μϕ = trainLearner(DDPG_actor01_critic0025[1],
#                     DDPG_actor01_critic0025[2]);

# replPlots(DDPG_actor01_critic0025[1].algorithm, 
#                     @getName(DDPG_actor01_critic0025), p)




# p, μϕ = trainLearner(DDPG_actor005_critic00125[1],
#                     DDPG_actor005_critic00125[2]);

# replPlots(DDPG_actor005_critic00125[1].algorithm, 
#                     @getName(DDPG_actor005_critic00125), p)





# p, μϕ = trainLearner(Learner(DDPG(),
#                 Online(),
#                 Clamped()),
#                 Parameter(environment="MountainCarContinuous-v0",
#                 train_start = 10000,
#                 max_episodes = 10,
#                 noise_type = "none",
#                 batch_size=128,
#                 η_actor = 0.001,
#                 η_critic = 0.001,
#                 τ_actor=0.1,
#                 τ_critic=0.025))


# display(replPlots(DDPG(), p))


# println("I have loaded.")

# p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
#                 Parameter(batch_size=1,
#                 batch_length=40,
#                 max_episodes_length=999,
#                 Sequences=200,
#                 dT = 0.01, 
#                 reward_hidden=[(32, 32), (32, 32)],
#                 dynode_hidden=[(20, 20), (20, 20)]));      


# display(replPlots(NODEModel(), p))

println("I am done, Good Bye")
# p, μϕ = trainLearner(Learner(DDPG(),
#                 Online(),
#                 Clamped()),
#                 Parameter(environment="MountainCarContinuous-v0",
#                 train_start = 10000,
#                 max_episodes = 200,
#                 noise_type = "none",
#                 batch_size=64,
#                 η_actor = 0.001,
#                 η_critic = 0.001,
#                 τ_actor=0.1,
#                 τ_critic=0.025));


# display(showResults(DDPG(), p, true))