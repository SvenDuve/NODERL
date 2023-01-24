using Distributed
#using UnicodePlots
using NODERL
using Hyperopt 
using Plots


macro getName(x)
    quote
        $(string(x))
    end
end

# set objects to iterate over here:



function objective(batch_length, dT, η_node, η_reward) 

    p, μϕ, Rϕ = trainLearner(Learner(DynaWorldModel(), Episodic(), Randomized()),
                        Parameter(environment="LunarLander-v2",
                                    batch_size=1,
                                    batch_length=batch_length,
                                    max_episodes_length=999,
                                    Sequences=20,
                                    dT = dT, 
                                    reward_hidden=[(32, 32), (32, 32)],
                                    dynode_hidden=[(20, 20), (20, 20)],
                                    η_node = η_node,
                                    η_reward = η_reward)); 

    # file = "lract" * string(η_actor) * "lrcr" * string(η_critic) * "taua" * string(τ_actor) * "tcr" = string(τ_critic)

    nms = ["batch_length", "dT", "η_node", "η_reward"]
    vls = [batch_length, dT, η_node, η_reward]
    file = string.([n * string(v) for (n, v) in zip(nms, vls)]...)
    
    storePlots(DynaWorldModel(), "LL_model" * file, p)
    #storeModel(μϕ, "output/" * file * ".bson", p)


    return sum(p.model_loss)


end

                               
ho = @phyperopt for i=20,
    sampler = RandomSampler(),     
    #episodes = StepRange(10, 1, 11),
    #batchsize = StepRange(32, 2, 36),
    batch_length = StepRange(10, 10, 40),
    dT = exp10.(LinRange(-3, -1, 5)),
    η_node = exp10.(LinRange(-4, -3, 10)), #learning rate actor
    η_reward = exp10.(LinRange(-4, -3, 10)) # learning rate critic
# delta = StepRange(10,5, 25),
# lr =  exp10.(LinRange(-4,-3,10)),
# mm =  LinRange(0.75,0.95,5),
# day0 = StepRange(5,3, 10)
    @show objective(batch_length, dT, η_node, η_reward)
end

open("output/LL_model.txt", "w") do io
    println(io, ho)
end


pl = plot(ho);

Plots.savefig(pl, "output/plot_LL_model.png")


best_params, min_f = ho.minimizer, ho.minimum

println("Best Parameters " * string(best_params))




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
