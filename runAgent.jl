using UnicodePlots
using NODERL

p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="MountainCarContinuous-v0",
                train_start = 10000,
                max_episodes = 10,
                noise_type = "none",
                batch_size=128,
                η_actor = 0.001,
                η_critic = 0.001,
                τ_actor=0.1,
                τ_critic=0.025))


display(replPlots(DDPG(), p))


println("I have loaded.")

p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
                Parameter(batch_size=1,
                batch_length=40,
                max_episodes_length=999,
                Sequences=200,
                dT = 0.01, 
                reward_hidden=[(32, 32), (32, 32)],
                dynode_hidden=[(20, 20), (20, 20)]));      


display(replPlots(NODEModel(), p))

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