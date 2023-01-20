#function loss(t::Critic, Y::Matrix{Float64}, S::Matrix{Float32}, A::Matrix{Float64})
function loss(t::Critic, Y, S, A)
    V = Qθ(vcat(S, A))
    (V .- Y) .^ 2 |> sum
end


function loss(t::Actor, S::Matrix{Float32})
    Qθ(vcat(S, μϕ(S))) |> sum |> mean
end


function loss(t::Reward, S, A, R, S′) 

    R̂ = Rϕ(vcat(S, A))

    return Flux.mse(R̂, R)

end



function loss(t::NODE, S, A, R, S′) 
    
    Ŝ = fθ(vcat(S, A))
    return Flux.mse(Ŝ, S′)

end

# function loss(t::NODE, S, A, R, S′) 
    
#     dS = fθ(vcat(S, A))

#     return Flux.mse(S′ - S, dS)

# end




function loss(t::DyNODE,  S, A, R, S′) 

    Ŝ = Zygote.Buffer(S)
    Ŝ = transition(t, S, A, R)[1]

    return (1 / p.state_size) * (1 / p.batch_length) * sum(abs.(copy(Ŝ) - S′))
    #return Flux.mse(Ŝ, S′)
    # return sum(abs.(copy(Ŝ) - S′))

end


function loss(t::DyReward, S, A, R, S′) 
    
    R̂ = Zygote.Buffer(R)
    R̂ = transition(t, S, A, R)[2]

    return Flux.mse(copy(R̂), R)

end




function transition(t::T, s, a, r) where T <: TransitionType

    ŝ = Zygote.Buffer(s)
    r̂ = Zygote.Buffer(r)
    state = s[:,1]
    for i in collect(1:p.batch_length)
        # @show size(vcat(state, a[:,i]))
        ds = fθ(vcat(state, a[:,i]))
        state = state + ds
        ŝ[:,i] = state
        r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
    end
    
    return copy(ŝ), copy(r̂)

end



function transitionValidation(t::T, s, a, r) where T <: TransitionType

    ŝ = similar(s)
    r̂ = similar(r)
    state = s[:,1]
    for i in collect(1:p.batch_length)
        ds = fθ(vcat(state, a[:,i]))
        state = state + ds
        ŝ[:,i] = state
        r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
    end
    
    return ŝ, r̂

end




# function transition(t::T, s, a, r) where T <: TransitionType

#     ŝ = Zygote.Buffer(s)
#     r̂ = Zygote.Buffer(r)
#     state = s[:,1]
#     for i in collect(1:length(a))
#         state = fθ(vcat(state, a[:,i]))
#         ŝ[:,i] = state
#         r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
#     end
    
#     return copy(ŝ), copy(r̂)

# end




# function lossReward(m::DyNodeModel, s, a, r, s′)
#     r̂ = [Rϕ(vcat(vcat(s′[:, k]...), action)) for (k, action) in enumerate(a)]
#     reward_loss = Flux.mse(hcat(r̂...), r)
#     return reward_loss
# end


# function rewardLoss(m::DyNodeModel, S, A, R, S′)

#     R̂ = Zygote.Buffer(R)
#     R̂ = transition(S, A, R)[2]

#     return Flux.mse(copy(R̂), R)

# end


# function dyNodeLoss(m::DyNodeModel, S, A, R, S′)

#     Ŝ = Zygote.Buffer(S)

#     Ŝ = transition(S, A, R)[1]

#     return (1 / p.state_size) * (1 / p.batch_length) * sum(abs.(copy(Ŝ) - S′))
#     #return Flux.mse(Ŝ, S′)
#     # return sum(abs.(copy(Ŝ) - S′))

# end



# function rewardLoss(m::NodeModel, S, A, R, S′)

#     R̂ = Rϕ(vcat(S, A))

#     return Flux.mse(R̂, R)

# end

# function rewardLoss(m::DynaWorldModel, S, A, R, S′)

#     R̂ = Rϕ(vcat(S, A))

#     return Flux.mse(R̂, R)

# end




# function nodeLoss(m::NodeModel, S, A, R, S′)

#     Ŝ = fθ(vcat(S, A))

#     return Flux.mse(Ŝ, S′)

# end





# function transition(m::DyNodeModel, s, a, r)

#     ŝ = Zygote.Buffer(s)
#     r̂ = Zygote.Buffer(r)
#     state = s[:,1]
#     for i in collect(1:length(a))
#         state = euler(state, a[:,i])
#         ŝ[:,i] = state
#         r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
#     end
    
#     return copy(ŝ), copy(r̂)
# end


# function transition(m::DynaWorldModel, s, a, r)

#     ŝ = Zygote.Buffer(s)
#     r̂ = Zygote.Buffer(r)
#     state = s[:,1]
#     for i in collect(1:length(a))
#         state = fθ(vcat(state, a[:,i]))
#         ŝ[:,i] = state
#         r̂[:,i] = Rϕ(vcat(s[:,i], a[:,i]))
#     end
    
#     return copy(ŝ), copy(r̂)
# end





# function euler(s, a)
#     x = vcat(s, a)
#     return s + p.dT * fθ(x)
# end   