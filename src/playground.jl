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

# wgt_act0.09444wgt_cr0.15778lr_no0.001lr_re0.00022lr_act0.0006lr_cr0.001

pms = Parameter(environment="LunarLander-v2",
                batch_size_episodic=1,
                batch_size=128,
                batch_length=40,
                noise_type="none",
                train_start = 0,
                max_episodes = 200,
                max_episodes_length=400,
                Sequences=400, #200
                model_episode_retrain = 20,
                dT = 0.01,
                η_node = 0.001,
                η_reward = 0.0001, #0.0002
                η_actor = 0.0004,
                η_critic = 0.001,
                τ_actor=0.12,
                τ_critic=0.01,
                reward_hidden=[(32, 32)],
                dynode_hidden=[(64, 64)],
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]); 



p, μϕ = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
            Learner(DDPG(), Online(), Clamped()), 
            pms);


showResults(DDPG(), p)

#38300
#40933
p.env_steps


using Conda
using PyCall


gym = pyimport("gym")
global env = gym.make(p.environment)


s = env.reset()
R = []
notSolved = true

while notSolved
    
    a = NODERL.action(Clamped(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    # a = NODERL.action(Randomized(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    
    s′, r, t, _ = env.step(a)
    append!(R, r)
    env.render()
    sleep(0.1)
    s = s′
    notSolved = !t
end

env.close()


pms = Parameter(environment="BipedalWalker-v3",
                batch_size_episodic=1,
                batch_size=128,
                batch_length=40,
                noise_type="none",
                train_start = 0,
                max_episodes = 200,
                max_episodes_length=1000,
                Sequences=600, #200
                model_episode_retrain = 50,
                dT = 0.01,
                η_node = 0.001,
                η_reward = 0.0001, #0.0002
                η_actor = 0.0004,
                η_critic = 0.001,
                τ_actor=0.12,
                τ_critic=0.01,
                reward_hidden=[(32, 32)],
                dynode_hidden=[(64, 64)],
                critic_hidden = [(16, 16)],
                actor_hidden = [(48, 48)]); 



p, μϕ = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
            Learner(DDPG(), Online(), Clamped()), 
            pms);


showResults(DDPG(), p)



pms = Parameter(environment="Pendulum-v1",
                batch_size_episodic=1,
                batch_size=128,
                batch_length=40,
                noise_type="gaussian",
                train_start = 0,
                max_episodes = 200,
                max_episodes_length=999,
                Sequences=200, #200
                model_episode_retrain = 10,
                dT = 0.01,
                η_node = 0.001,
                η_reward = 0.00022, #0.0002
                η_actor = 0.0006,
                η_critic = 0.001,
                τ_actor=0.095,
                τ_critic=0.16,
                reward_hidden=[(64, 64)],
                dynode_hidden=[(40, 40), (40, 40)],
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]); 



p, μϕ = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
            Learner(DDPG(), Online(), Clamped()), 
            pms);

#38300
#40933
p.env_steps

using Conda
using PyCall
gym = pyimport("gym")
global env = gym.make(p.environment)


s = env.reset()
R = []
notSolved = true

while notSolved

    a = NODERL.action(Clamped(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    # a = NODERL.action(Randomized(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)

    s′, r, t, _ = env.step(a)
    append!(R, r)
    env.render()
    sleep(0.1)
    s = s′
    notSolved = !t
end

env.close()



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




q, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="MountainCarContinuous-v0",
                train_start = 1000,
                max_episodes = 200,
                noise_type = "none",
                batch_size=128,
                η_actor = 0.001,
                η_critic = 0.001,
                τ_actor=0.1,
                τ_critic=0.025,
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]));


showResults(DDPG(), q)


p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                Parameter(environment="Pendulum-v1",
                train_start = 1000,
                max_episodes = 200,
                # critic_hidden = [(64, 128), (128, 64)],
                # actor_hidden = [(64, 128), (128, 64)],
                noise_type = "gaussian",
                batch_size=128,
                η_actor = 0.0001,
                η_critic = 0.0001,
                τ_actor=0.0025,
                τ_critic=0.0025,
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]));


showResults(DDPG(), p)


p, μϕ = trainLearner(Learner(DDPG(),
                Online(),
                Clamped()),
                # Parameter(environment="MountainCarContinuous-v0",
                Parameter(environment="LunarLander-v2",
                train_start = 1000,
                max_episodes = 600,
                noise_type = "none",
                batch_size=128,
                η_actor = 0.0001,
                η_critic = 0.0001,
                τ_actor=0.00125,
                τ_critic=0.00625,
                # η_actor = 0.0001,
                # η_critic = 0.0001,
                # τ_actor=0.1,
                # τ_critic=0.025,
                critic_hidden = [(64, 64)],
                actor_hidden = [(128, 128)]));
                # critic_hidden = [(32, 32)],
                # actor_hidden = [(64, 64)]))

# 48283
showResults(DDPG(), p)





p, fθ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()), 
                # Parameter(environment="LunarLander-v2",
                Parameter(environment="MountainCarContinuous-v0",
                # Parameter(environment="LunarLander-v2",
                batch_size_episodic=1, #64
                batch_length=40,
                noise_type="none",
                max_episodes_length=999,
                Sequences=100, #200
                dT = 0.01,
                η_node = 0.0045,
                η_reward = 0.001, #0.0002
                reward_hidden=[(128, 128)],
                dynode_hidden=[(128, 128)]));        


using Plots
using Statistics



meanModelLoss = [mean(p.model_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
meanRewardLoss = [mean(p.reward_loss[i-9:i]) for i in collect(10:length(p.reward_loss))]


plot(meanModelLoss, c=:red, label="Model Loss")
plot!(collect(range(1,591,60)), p.validation_loss, c=:blue, label="Model Val Loss")
plot!(twinx(), meanRewardLoss, c=:green, label="Reward Loss")






showLoss(p)





using Conda
using PyCall

p = Parameter()

gym = pyimport("gym")
global env = gym.make(p.environment)
p = resetParameters(p)


s = env.reset()
R = []
notSolved = true

while notSolved

    a = NODERL.action(Clamped(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)
    # a = NODERL.action(Randomized(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)

    s′, r, t, _ = env.step(a)
    append!(R, r)
    env.render()
    sleep(0.1)
    s = s′
    notSolved = !t
end

env.close()

μϕ





agentTraineModel



# p, fθ, Rϕ = agentTrainedModel(Learner(DynaWorldModel(), Episodic(), Clamped()), 
#                 # Parameter(environment="Pendulum-v1",
#                 Parameter(environment="MountainCarContinuous-v0",
#                 batch_size_episodic=1, #64
#                 batch_length=40,
#                 noise_type="none",
#                 max_episodes_length=999,
#                 Sequences=400, #200
#                 dT = 0.01,
#                 η_node = 0.00045,
#                 η_reward = 0.001, #0.0002
#                 reward_hidden=[(64, 64)],
#                 dynode_hidden=[(32, 32)]), μϕ);        



using Statistics
using Plots

meanModelLoss = [mean(p.model_loss[i-9:i]) for i in collect(10:length(p.model_loss))]
meanRewardLoss = [mean(p.reward_loss[i-9:i]) for i in collect(10:length(p.reward_loss))]


plot(meanModelLoss, c=:red, label="Model Loss")
plot!(collect(range(1,391,40)), p.validation_loss, c=:blue, label="Model Val Loss")
plot!(twinx(), meanRewardLoss, c=:green, label="Reward Loss")
                



using Distributions

S = []
A = []
S′ = []
R = []

s = env.reset()

for i in 1:10000

    a = [rand(Uniform(el[1], el[2])) for el in zip(p.action_bound_low, p.action_bound_high)] 
    s′ = s .+ 0.01 .* fθ(vcat(s, a))
    r = Rϕ(vcat(s, a))
    # randS = [rand(Uniform(el[1], el[2])) for el in zip(p.state_low, p.state_high)]
    # randA = [rand(Uniform(el[1], el[2])) for el in zip(p.action_bound_low, p.action_bound_high)]
    append!(S, [s])
    append!(A, a)
    append!(S′, [s′])
    append!(R, r)    
    s = s′

end


S


scatter(hcat(S...)[1,:], hcat(S...)[2,:])
plot(R)



p, μϕ = modelTrainedAgent(Learner(DDPG(),
                    Online(),
                    Clamped()),
                    Parameter(environment="MountainCarContinuous-v0",
                    train_start = 1000,
                    max_episodes = 200,
                    # critic_hidden = [(64, 128), (128, 64)],
                    # actor_hidden = [(64, 128), (128, 64)],
                    noise_type = "none",
                    batch_size=128,
                    η_actor = 0.001,
                    η_critic = 0.0002,
                    τ_actor=0.1,
                    τ_critic=0.2,
                    critic_hidden = [(32, 32), (32, 32)],
                    actor_hidden = [(32, 32), (32, 32)]), fθ, Rϕ) 


#871



# Env_steps appr. 35k with perfect trained agent trained model trained agent

using Flux, Flux.Optimise
import Flux.params

Chain(Dense(p.state_size, p.actor_hidden[1][1]), BatchNorm(p.actor_hidden[1][1], relu),
                Chain([Dense(el[1], el[2], relu) for el in p.actor_hidden]...),
                Dense(p.actor_hidden[end][2], p.action_size, tanh),
                x -> x * p.action_bound)



Chain(Dense(p.state_size, p.actor_hidden[1][1]), BatchNorm(p.actor_hidden[1][1], relu), 
    Chain(
        Dense(p.actor_hidden[1][1], p.actor_hidden[1][2]),
        BatchNorm(p.actor_hidden[1][2], relu)))



m = Chain(
    Dense(28^2, 64),
    BatchNorm(64, relu),
    Dense(64, 10),
    BatchNorm(10),
    softmax)