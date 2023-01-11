# Raspberry

# wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.8/julia-1.8.4-linux-aarch64.tar.gz
# tar zxvf julia-1.8.4-linux-aarch64.tar.gz
# export PATH="$PATH:/home/sven/julia-1.8.4/julia/bin"

# machine popos
# name: 192.168.178.136
# user: svenduve
# pass: pilsener



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





# batch_length        dT    η_node  η_reward 
# 40  0.003162     0.001 0.0002154 


# Parameter(batch_size=1,
#                 batch_length=40,
#                 max_episodes_length=999,
#                 Sequences=200,
#                 dT = 0.003,
#                 η_node = 0.001,
#                 η_reward = 0.0002, 
#                 reward_hidden=[(32, 32)],
#                 dynode_hidden=[(32, 32)]));    



p, buffer_one, buffer_two = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
            Learner(DDPG(), Online(), Clamped()), 
            Parameter(environment="MountainCarContinuous-v0",
            train_start = 1000,
            max_episodes = 100,
            max_episodes_length = 999,
            max_episodes_length_mb = 998,
            Sequences = 100, # DynaWorld Learning #100
            dT=0.003,
            η_node=0.001,
            η_reward = 0.0002,
            trainloops_mb = 100, # model based learner #10
            batch_size_episodic=1,
            batch_size=64,
            reward_hidden=[(32, 32), (32, 32)],
            dynode_hidden=[(32, 32), (32, 32)],
            η_actor = 0.001,
            η_critic = 0.001,
            τ_actor=0.1,
            τ_critic=0.025));



p, μϕ = trainLearner(DDPG_actor01_critic0025[1],
                    DDPG_actor01_critic0025[2])

replPlots(DDPG_actor01_critic0025[1].algorithm, 
            @getName(DDPG_actor01_critic0025), p)



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
                Parameter(environment="MountainCarContinuous-v0",
                train_start = 10000,
                max_episodes = 200,
                # critic_hidden = [(64, 128), (128, 64)],
                # actor_hidden = [(64, 128), (128, 64)],
                noise_type = "none",
                batch_size=128,
                η_actor = 0.0005,
                η_critic = 0.0002,
                τ_actor=0.01,
                τ_critic=0.2,
                critic_hidden = [(32, 32), (32, 32)],
                actor_hidden = [(32, 32), (32, 32)]))


showResults(DDPG(), p)



p, fθ, Rϕ = trainLearner(Learner(NODEModel(), Episodic(), Randomized()), 
                Parameter(batch_size=128,
                max_episodes_length=999,
                Sequences=1000,
                dT = 0.003, 
                η_node = 0.001,
                η_reward = 0.0002,        
                reward_hidden=[(32, 32)],
                dynode_hidden=[(32, 32)]));




using Plots
using Statistics
meanModelLoss = [mean(p.model_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
meanRewardLoss = [mean(p.reward_loss[i-9:i]) for i in collect(10:length(p.reward_loss))]
                
                
plot(meanModelLoss, c=:red)
plot!(twinx(), meanRewardLoss, c=:green)




p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
                #Parameter(environment="Pendulum-v1",
                Parameter(environment="MountainCarContinuous-v0",
                batch_size_episodic=1, #64
                batch_length=10,
                max_episodes_length=999,
                Sequences=200, #200
                dT = 0.03,
                η_node = 0.0005,
                η_reward = 0.0002, #0.0002
                reward_hidden=[(32, 32)],
                dynode_hidden=[(32, 32)]));        


using Plots
using Statistics



meanModelLoss = [mean(p.model_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
meanRewardLoss = [mean(p.reward_loss[i-9:i]) for i in collect(10:length(p.reward_loss))]


plot(meanModelLoss, c=:red, label="Model Loss")
plot!(collect(range(1,191,20)), p.validation_loss, c=:blue, label="Model Val Loss")
plot!(twinx(), meanRewardLoss, c=:green, label="Reward Loss")






showLoss(p)


