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
τ_critic=0.025))



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
                
                






function objective(episodes, batchsize) 

    p, μϕ = trainLearner(Learner(DDPG(),
    Online(),
    Clamped()),
    Parameter(environment="MountainCarContinuous-v0",
    train_start = 10000,
    max_episodes = episodes,
    noise_type = "none",
    batch_size=batchsize,
    η_actor = 0.001,
    η_critic = 0.001,
    τ_actor=0.1,
    τ_critic=0.025));

    return -sum(p.total_rewards[end-2, end])


end

                
using Hyperopt                
ho = @hyperopt for i=3,
    sampler = RandomSampler(),     
    episodes = StepRange(10, 1, 11),
    batchsize = StepRange(32, 2, 36)
# delta = StepRange(10,5, 25),
# lr =  exp10.(LinRange(-4,-3,10)),
# mm =  LinRange(0.75,0.95,5),
# day0 = StepRange(5,3, 10)
    @show objective(episodes, batchsize)
end
# best_params, min_f = ho.minimizer, ho.minimum


# import gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = policy(observation)  # User-defined policy function
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()


using Conda
using PyCall


gym = pyimport("gym")
env = gym.make("LunarLander-v2"; render_mode="human")



py"""
import gym
def print_one_number(mode):  
    env = gym.make("LunarLander-v2", continuous=mode)
    s = env.reset()
    return s
"""
print_one_number_py = py"print_one_number"  # only the 1-line version of the py macro yields the return value back to Julia!
mode = true
s = print_one_number_py(mode)