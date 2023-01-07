
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




function MBDDPGAgent(model::Learner, agent::Learner, pms::Parameter) 

    gym = pyimport("gym")
    global world = gym.make(pms.environment)

    p, fθ, Rϕ = trainLearner(model, pms)

    setFunctionApproximation(DDPG())


    println("done initial Model Training")

    global 𝒟_RL = []

    for j in 1:p.trainloops_mb
        println("MPC round $j")
        episodeRewards = []
        s = world.reset()
        
        for i in 1:p.max_episodes_length_mb

            a = action(MPC(), model.train, s, p)
            p.env_steps += 1
            # a = action(Randomized(), model.train, s, p)
            s′, r, t, _ = world.step(a)
            # @show s′, r, t
            append!(episodeRewards, r)
            remember(MPCBuffer(), p.mem_size, s, a, r, s′, t)
            
            s = s′

            if t
               return
            end



        end

        if size(𝒟_RL)[1] > p.DDPG_batch # try and sat a parameter
            initTrainDDPG(modelDDPG())
            println("Trained some DDPG.")
        end

        append!(p.world_rewards, sum(episodeRewards))
        println("Retraining...")
        retrain(DynaWorldModel(), model)
        println("Retraining done.")
    end

    # @show p.world_rewards

    return (p, 𝒟, 𝒟_RL)

end

# function dyNode(m::DyNodeModel, pms::Parameter)

#     # To Do's:
#     # to set up dynode_batch_size -> 64 in the paper

#     # interactions with the real World


#     gym = pyimport("gym")
#     global env = gym.make(pms.environment)
#     global p = resetParameters(pms)

#     #setNoise(p)

#     # set buffer
#     global 𝒟 = []

#     # global fθ = setNode(m, p)
#     global fθ = setNetwork(m) # Code up a Network that will be solved with euler steps
#     global Rϕ = setNetwork(Rewards())


#     for i in 1:p.Sequences
#         ep = Episode(env, m, p)()
#         for (s, a, r, s′, t) in ep.episode
#             remember(p.mem_size, s, a, r, s′, t)
#         end

#         model_loss, reward_loss = train(m)
#         # alt_train(m)
#         if i % 10 == 0
#             println("Iteration $i")
#         end
#         append!(p.model_loss, model_loss)
#         append!(p.reward_loss, reward_loss)
#     end
#     return p, fθ, Rϕ
# end





# function NODEAgent(m::NodeModel, pms::Parameter)

#     # To Do's:
#     # to set up dynode_batch_size -> 64 in the paper

#     # interactions with the real World


#     gym = pyimport("gym")
#     global env = gym.make(pms.environment)
#     global p = resetParameters(pms)

#     #setNoise(p)
#     #@show noise

#     # set buffer
#     global 𝒟 = []

#     global fθ = setNode(m, p)
#     global Rϕ = setNetwork(Rewards())


#     for i in 1:p.Sequences
#         ep = Episode(env, m, p)()
#         for (s, a, r, s′, t) in ep.episode
#             remember(p.mem_size, s, a, r, s′, t)
#         end

#         model_loss, reward_loss = train(m)
#         if i % 10 == 0
#             println("Iteration $i")
#         end
#         append!(p.model_loss, model_loss)
#         append!(p.reward_loss, reward_loss)
#     end
    
#     return p, fθ, Rϕ

# end





# function dynaWorld(m::DynaWorldModel, pms::Parameter)

#     # To Do's:
#     # to set up dynode_batch_size -> 64 in the paper

#     # interactions with the real World


#     gym = pyimport("gym")
#     global env = gym.make(pms.environment)
#     global p = resetParameters(pms)

#     #setNoise(p)
#     #@show noise

#     # set buffer
#     global 𝒟 = []

#     global fθ = setNode(m, p)
#     global Rϕ = setNetwork(Rewards())


#     for i in 1:p.Sequences
#         ep = Episode(env, m, p)()
#         for (s, a, r, s′, t) in ep.episode
#             remember(p.mem_size, s, a, r, s′, t)
#         end

#         model_loss, reward_loss = train(m)
#         if i % 10 == 0
#             println("Iteration $i")
#         end
#         append!(p.model_loss, model_loss)
#         append!(p.reward_loss, reward_loss)
#     end
    
#     return p, fθ, Rϕ

# end
