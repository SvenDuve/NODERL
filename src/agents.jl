
function trainLearner(l::Learner, pms::Parameter)

    println("Welcome to this function.")
    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)
    
    setNoise(p)

    global 𝒟 = []

    setFunctionApproximation(l.algorithm)

    train(l.algorithm, l)

    return getVals(l.algorithm)

end #trainAgent



function agentTrainedModel(l::Learner, pms::Parameter, policy)

    println("Welcome to this function.")
    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)
    global μϕ = policy
    
    setNoise(p)

    global 𝒟 = []

    setFunctionApproximation(l.algorithm)

    train(l.algorithm, l)

    return getVals(l.algorithm)


end


function modelTrainedAgent(l::Learner, pms::Parameter, model, reward) 

    println("Welcome to this function.")
    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)

    fθ, Rϕ = model, reward 

    setNoise(p)

    #global 𝒟 = [] # Question, should we separate model and real experience

    setFunctionApproximation(l.algorithm)

    trainOnModel(l.algorithm, l)

    return getVals(l.algorithm)

end








function MBDDPGAgent(model::Learner, agent::Learner, pms::Parameter) 

    gym = pyimport("gym")
    global world = gym.make(pms.environment)

    p, fθ, Rϕ = trainLearner(model, pms)

    p, μϕ = modelTrainedAgent(agent, pms, fθ, Rϕ)

    return p, μϕ

end




# function MBDDPGAgent(model::Learner, agent::Learner, pms::Parameter) 

#     gym = pyimport("gym")
#     global world = gym.make(pms.environment)

#     p, fθ, Rϕ = trainLearner(model, pms)

#     setFunctionApproximation(DDPG())


#     println("done initial Model Training")

#     global 𝒟_RL = []

#     for j in collect(1:p.trainloops_mb)

#         println("MPC round $j")
#         episodeRewards = []
#         s::Vector{Float32} = world.reset()
#         r::Float64 = 0.0
#         a::Vector{Float32} = [0.0] # check action space
#         t::Bool = false
        
#         for i in 1:p.max_episodes_length_mb

#             a = action(MPC(), model.train, s, p)
#             p.env_steps += 1
#             # a = action(Randomized(), model.train, s, p)
#             s′, r, t, _ = world.step(a)
#             # @show s′, r, t
#             append!(episodeRewards, r)
#             remember(MPCBuffer(), p.mem_size, s, a, r, s′, t)
            
#             s = s′

#             if t
#                return
#             end

#         end

#         if size(𝒟_RL)[1] > p.DDPG_batch # try and set a parameter
#             trainDDPG(modelDDPG())
#             println("Trained some DDPG.")
#         end

#         append!(p.world_rewards, sum(episodeRewards))
#         println("Retraining...")
#         retrain(DynaWorldModel(), model)
#         println("Retraining done.")
#     end


#     for j in collect(1:p.max_episodes)

#         println("DDPG round $j")
#         episodeRewards = []
#         s::Vector{Float32} = world.reset()
#         r::Float64 = 0.0
#         a::Vector{Float32} = [0.0] # check action space
#         t::Bool = false


#         for k in collect(1:800)

#             a = action(Clamped(), model.train, s, p)
#             p.env_steps += 1
#             # a = action(Randomized(), model.train, s, p)
#             s′, r, t, _ = world.step(a)
#             # @show s′, r, t
#             append!(episodeRewards, r)
#             remember(MPCBuffer(), p.mem_size, s, a, r, s′, t)
#             trainDDPG(modelDDPG())
            
#             s = s′

#             if t
#                 return
#             end

#         end

#         println("Episode Rewards: $(sum(episodeRewards))")
#         append!(p.world_rewards, sum(episodeRewards))

#     end

#     # @show p.world_rewards

#     return (p, 𝒟, 𝒟_RL)

# end

