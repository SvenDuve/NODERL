using UnicodePlots
using NODERL
using Hyperopt 


macro NameVal(arg)
    string.(arg) .* string.(eval.(arg))
end
# function nameVal(arg...)
#     string.(Symbol(arg)) .* string.(eval.(arg))
# end


function objective(τ_actor, τ_critic, η_node, η_reward, η_actor, η_critic) 


    pms = Parameter(environment="MountainCarContinuous-v0",
                batch_size_episodic=1,
                batch_size=128,
                batch_length=40,
                noise_type="none",
                train_start = 1000,
                max_episodes = 200, #200,
                max_episodes_length=999,
                Sequences=600,# 600, #200
                dT = 0.01,
                η_node = η_node,
                η_reward = η_reward, #0.0002
                η_actor = η_actor,
                η_critic = η_critic,
                τ_actor=τ_actor,
                τ_critic=τ_critic,
                reward_hidden=[(64, 64)],
                dynode_hidden=[(40, 40), (40, 40)],
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]) 


    p, μϕ = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
    Learner(DDPG(), Online(), Clamped()), 
    pms)


    nms = ["wgt_act", "wgt_cr", "lr_no", "lr_re", "lr_act", "lr_cr"]
    vls = [τ_actor, τ_critic, η_node, η_reward, η_actor, η_critic]
    file = string.([n * string(round(v, digits=5)) for (n, v) in zip(nms, vls)]...)
    
    replPlots(DDPG(), file, p)
    storeModel(μϕ, "output/" * file * ".bson", p)

    return sum(p.env_steps)


end


ho = @hyperopt for i=20,
    sampler = RandomSampler(),     
    #episodes = StepRange(10, 1, 11),
    #batchsize = StepRange(32, 2, 36),
    τ_actor = LinRange(0.01, 0.2, 10), #weight cp params actor 
    τ_critic = LinRange(0.01, 0.2, 10), # weight cp params actor
    η_node = exp10.(LinRange(-4, -3, 10)), # Learning rate Node
    η_reward = exp10.(LinRange(-4, -3, 10)), # Learning rate Reward
    η_actor = exp10.(LinRange(-4, -3, 10)), #learning rate actor
    η_critic = exp10.(LinRange(-4, -3, 10)) # learning rate critic
# delta = StepRange(10,5, 25),
# lr =  exp10.(LinRange(-4,-3,10)),
# mm =  LinRange(0.75,0.95,5),
# day0 = StepRange(5,3, 10)
    @show objective(τ_actor, τ_critic, η_node, η_reward, η_actor, η_critic)
end

open("output/result.txt", "w") do io
    println(io, ho)
end


best_params, min_f = ho.minimizer, ho.minimum

println("Best Parameters " * string(best_params))

