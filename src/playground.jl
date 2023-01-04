# Raspberry

# machine name: learner-one
# user: sven 
# pass: pilsener

# machine name: learner-two 
# user: sven
# pass: pilsener



# Improve
# perhaps reduce one train function
# check type hierarchy
# introdue annealing


# environments:

#environment="Pendulum-v1"
#environment="BipedalWalker-v3"




p, μϕ = trainLearner(Learner(DDPG(), 
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
                τ_critic=0.025))



p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="MountainCarContinuous-v0",
                train_start = 10000,
                max_episodes = 200,
                noise_type = "none",
                batch_size=128,
                η_actor = 0.001,
                η_critic = 0.001,
                τ_actor=0.1,
                τ_critic=0.025));


showResults(DDPG(), p)


p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="Pendulum-v1",
                train_start = 10000,
                max_episodes = 200,
                # critic_hidden = [(64, 128), (128, 64)],
                # actor_hidden = [(64, 128), (128, 64)],
                noise_type = "none",
                batch_size=128,
                η_actor = 0.001,
                η_critic = 0.001,
                τ_actor=0.1,
                τ_critic=0.025));


showResults(DDPG(), p)


p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="BipedalWalker-v3",
                train_start = 1000,
                max_episodes = 1000,
                # critic_hidden = [(64, 128), (128, 64)],
                # actor_hidden = [(64, 128), (128, 64)],
                noise_type = "none",
                batch_size=128,
                η_actor = 0.001,
                η_critic = 0.001,
                τ_actor=0.1,
                τ_critic=0.025))


showResults(DDPG(), p)



p, fθ, Rϕ = trainLearner(Learner(NODEModel(), Episodic(), Randomized()), 
                Parameter(batch_size=64,
                batch_length=40,
                max_episodes_length=999,
                Sequences=100,
                dT = 0.01, 
                reward_hidden=[(32, 64), (64, 32)],
                dynode_hidden=[(32, 64), (64, 32)]));        



p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
                Parameter(batch_size=1,
                batch_length=40,
                max_episodes_length=999,
                Sequences=200,
                dT = 0.01, 
                reward_hidden=[(32, 32), (32, 32)],
                dynode_hidden=[(10, 10), (10, 10)]));        


using Plots
showLoss(p)





meanModelLoss = [mean(p.model_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
meanRewardLoss = [mean(p.reward_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
                
                
plot(meanModelLoss, c=:red)
plot!(twinx(), meanRewardLoss, c=:green)
                



p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
                Parameter(batch_size=64,
                max_episodes_length=999,
                Sequences=300,
                dT = 0.01, 
                reward_hidden=[(32, 32), (32, 32)]));        
                
                


