
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
