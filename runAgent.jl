using UnicodePlots
using NODERL


macro getName(x)
    quote
        $(string(x))
    end
end

# set objects to iterate over here:


DDPG_actor01_critic0025 = (Learner(DDPG(), 
Online(), 
Clamped()), 
Parameter(environment="MountainCarContinuous-v0",
train_start = 10000,
max_episodes = 20,
max_episodes_length = 1000,
batch_size=128,
η_actor = 0.001,
η_critic = 0.001,
τ_actor=0.1,
τ_critic=0.025));

DDPG_actor005_critic00125 = (Learner(DDPG(), 
Online(), 
Clamped()), 
Parameter(environment="MountainCarContinuous-v0",
train_start = 10000,
max_episodes = 20,
max_episodes_length = 1000,
batch_size=128,
η_actor = 0.001,
η_critic = 0.001,
τ_actor=0.05,
τ_critic=0.0125));




p, μϕ = trainLearner(DDPG_actor01_critic0025[1],
                    DDPG_actor01_critic0025[2]);

replPlots(DDPG_actor01_critic0025[1].algorithm, 
                    @getName(DDPG_actor01_critic0025), p)




p, μϕ = trainLearner(DDPG_actor005_critic00125[1],
                    DDPG_actor005_critic00125[2]);

replPlots(DDPG_actor005_critic00125[1].algorithm, 
                    @getName(DDPG_actor005_critic00125), p)





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