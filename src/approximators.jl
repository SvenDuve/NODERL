function setFunctionApproximation(algorithm::DDPG)

    # set Critic
    global Qθ = setNetwork(Critic())
    global Qθ′ = deepcopy(Qθ)

    # get Actor

    global μϕ = setNetwork(Actor())
    global μϕ′ = deepcopy(μϕ)

    
end


function setFunctionApproximation(algorithm::T) where T <: NODEArchitecture 
    
    global fθ = setNetwork(NODE()) # Code up a Network that will be solved with euler steps
    global Rϕ = setNetwork(Reward())
    
end


# function setFunctionApproximation(algorithm::DynaWorldModel)

#     global fθ = setNetwork(DyNode()) # Code up a Network that will be solved with euler steps
#     global Rϕ = setNetwork(DyRewards())

# end



