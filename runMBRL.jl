using UnicodePlots
using NODERL
using Hyperopt 


macro NameVal(arg)
    string.(arg) .* string.(eval.(arg))
end
# function nameVal(arg...)
#     string.(Symbol(arg)) .* string.(eval.(arg))
# end


function objective(batchsize, sequence, retrain) 


    pms = Parameter(environment="MountainCarContinuous-v0",
                batch_size_episodic=1,
                batch_size=batchsize,
                batch_length=40,
                noise_type="none",
                train_start = 0,
                max_episodes = 200,
                max_episodes_length=999,
                Sequences=sequence, #200
                model_episode_retrain = retrain,
                dT = 0.01,
                η_node = 0.001,
                η_reward = 0.0001, #0.0002
                η_actor = 0.0004,
                η_critic = 0.001,
                τ_actor=0.12,
                τ_critic=0.01,
                reward_hidden=[(64, 64)],
                dynode_hidden=[(40, 40), (40, 40)],
                critic_hidden = [(32, 32)],
                actor_hidden = [(64, 64)]); 


    p, μϕ = MBDDPGAgent(Learner(DynaWorldModel(), Episodic(), Randomized()), 
    Learner(DDPG(), Online(), Clamped()), 
    pms)


    nms = ["batchsize", "sequence", "retrain"]
    vls = [batchsize, sequence, retrain]
    file = string.([n * string(v) for (n, v) in zip(nms, vls)]...)
    
    replPlots(DDPG(), file, p)
    storeModel(μϕ, "output/" * file * ".bson", p)

    return -(sum(p.total_rewards) / p.env_steps)


end


sampler = RandomSampler(),     
ho = @hyperopt for i=1,
    #episodes = StepRange(10, 1, 11),
    batchsize = StepRange(32, 32, 128),
    sequence = StepRange(50, 50, 400),
    retrain = StepRange(10, 10, 20)
    #τ_actor = LinRange(0.01, 0.2, 10), #weight cp params actor 
    #τ_critic = LinRange(0.01, 0.2, 10), # weight cp params actor
    #η_node = exp10.(LinRange(-4, -3, 10)), # Learning rate Node
    #η_reward = exp10.(LinRange(-4, -3, 10)), # Learning rate Reward
    #η_actor = exp10.(LinRange(-4, -3, 10)), #learning rate actor
    #η_critic = exp10.(LinRange(-4, -3, 10)) # learning rate critic
# delta = StepRange(10,5, 25),
# lr =  exp10.(LinRange(-4,-3,10)),
# mm =  LinRange(0.75,0.95,5),
# day0 = StepRange(5,3, 10)
    @show objective(batchsize, sequence, retrain)
end

open("output/result_MBRL_batchsize_sequence_retrain.txt", "w") do io
    println(io, ho)
end


best_params, min_f = ho.minimizer, ho.minimum

println("Best Parameters " * string(best_params))

