# function setNetwork(nn::Critic, p::Parameter)
function setNetwork(nn::Critic)
    return Chain(Dense(p.state_size + p.action_size, p.critic_hidden[1][1], relu),
                Chain([Dense(el[1], el[2], relu) for el in p.critic_hidden]...),
                Dense(p.critic_hidden[end][2], 1))

end

# issue here with the network creation, review loop

function setNetwork(nn::Actor)
    return Chain(Dense(p.state_size, p.actor_hidden[1][1], relu), 
                Chain([Dense(el[1], el[2], relu) for el in p.actor_hidden]...),
                Dense(p.actor_hidden[end][2], p.action_size, tanh),
                x -> x * p.action_bound)

end



function setNetwork(nn::Reward)

    return Chain(Dense(p.state_size + p.action_size, p.reward_hidden[1][1], relu),
                Chain([Dense(el[1], el[2], relu) for el in p.reward_hidden]...),
                # BatchNorm(p.reward_hidden[1][1], relu),
                # Chain([Chain(Dense(el[1], el[2]), BatchNorm(el[2], relu)) for el in p.reward_hidden]...),
                # Dense(p.reward_hidden[end][2], 1, tanh))
                Dense(p.reward_hidden[end][2], 1))

end


function setNetwork(nn::T) where T <: NODEArchitecture

    down = Dense(p.state_size + p.action_size, p.dynode_hidden[1][1])

    dudt = Chain([Dense(el[1], el[2], elu) for el in p.dynode_hidden]...)
        
#    nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(),
    nn_ode = NeuralODE(dudt, (0.0f0, p.dT), Tsit5(),
        save_everystep=false,
        reltol=1e-3, abstol=1e-3,
        save_start=false) #|> gpu

    fc = Dense(p.dynode_hidden[end][2], p.state_size)

    return Flux.Chain(down, nn_ode, first, fc)

end


# function setNetwork(nn::DyNodeModel) # Generate a NN to be solved with Euler updates

#     return Chain(Dense(p.state_size + p.action_size, p.dynode_hidden[1][1], elu),
#                 Chain([Dense(el[1], el[2], elu) for el in p.dynode_hidden]...),
#                 Dense(p.dynode_hidden[end][2], p.state_size))


# end

