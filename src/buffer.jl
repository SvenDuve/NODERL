

function remember(m::GeneralBuffer, mem_size, s::Vector{Float32}, a::Vector{Float32}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟) >= mem_size
        deleteat!(𝒟, 1)
    end
    push!(𝒟, [s, a, r, s′, t])
end #remember

function remember(m::WorldBuffer, mem_size, s::Vector{Float32}, a::Vector{Float32}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟_World) >= mem_size
        deleteat!(𝒟_World, 1)
    end
    push!(𝒟_World, [s, a, r, s′, t])
end #remember


# function remember(m::MPCBuffer, mem_size, s::Vector{Float32}, a::Vector{Float32}, r::Float64, s′::Vector{Float32}, t::Bool)
#     if length(𝒟_RL) >= mem_size
#         deleteat!(𝒟_RL, 1)
#     end
#     # reduce first element of p.episode_length by one
#     # check if first element reached 0, then pop
#     push!(𝒟_RL, [s, a, r, s′, t])
# end #remember


function sampleBuffer(m::Process)
    minibatch = sample(𝒟, p.batch_size)
    X = hcat(minibatch...)
    S = hcat(X[1, :]...)
    A = hcat(X[2, :]...)
    R = hcat(X[3, :]...)
    S′ = hcat(X[4, :]...)
    T = hcat(X[5,:]...)
    return (S, A, R, S′, T)
end #sampleBuffer


function sampleBuffer(m::modelDDPG)
    minibatch = sample(𝒟_RL, p.batch_size)
    X = hcat(minibatch...)
    S = hcat(X[1, :]...)
    A = hcat(X[2, :]...)
    R = hcat(X[3, :]...)
    S′ = hcat(X[4, :]...)
    return (S, A, R, S′)
end #sampleBuffer



# function sampleBuffer(m::Episodic)

#     numEps = size(𝒟)[1] ÷ p.max_episodes_length #fails when max epiosde length exceeds the size
#     transInds = vcat([collect(((i-1)*p.max_episodes_length+1:i*p.max_episodes_length-p.batch_length)) for i in 1:Int(numEps)]...)
#     indStart = sample(transInds, p.batch_size) # to set up dynode_batch_size -> 64 in the paper
#     slices = [collect(i:i+p.batch_length-1) for i in indStart]

#     # the parameter to loop is S[:,:,i], this returns a matrix of an episode
#     batch = 𝒟[vcat(slices...)]
#     X = hcat(batch...)
#     S = reshape(hcat(X[1, :]...), (p.state_size, p.batch_length, p.batch_size)) #reshape(hcat(X[1,:]...), (2, 40, 128))
#     A = reshape(hcat(X[2, :]...), (p.action_size, p.batch_length, p.batch_size))
#     R = reshape(hcat(X[3, :]...), (1, p.batch_length, p.batch_size))
#     S′ = reshape(hcat(X[4, :]...), (p.state_size, p.batch_length, p.batch_size))
#     return (S, A, R, S′)
# end #sampleBuffer


function sampleBuffer(m::Episodic)
    # @show 𝒟
    # @show size(𝒟) p.episode_length p.batch_size_episodic
    # @show p.batch_length

    start_Points = 1 .+ vcat(0, cumsum(p.episode_length)[1:end-1])
    end_Points = cumsum(p.episode_length)

    # @show start_Points end_Points

    transInds = vcat([collect(el[1]:(el[2]-p.batch_length)+1) for el in zip(start_Points, end_Points)]...)
    indStart = sample(transInds, p.batch_size_episodic) # to set up dynode_batch_size -> 64 in the paper
    slices = [collect(i:i+p.batch_length-1) for i in indStart]

    # @show size(slices)

    # the parameter to loop is S[:,:,i], this returns a matrix of an episode
    batch = 𝒟[vcat(slices...)]
    X = hcat(batch...)
    S = reshape(hcat(X[1, :]...), (p.state_size, p.batch_length, p.batch_size_episodic)) #reshape(hcat(X[1,:]...), (2, 40, 128))
    A = reshape(hcat(X[2, :]...), (p.action_size, p.batch_length, p.batch_size_episodic))
    R = reshape(hcat(X[3, :]...), (1, p.batch_length, p.batch_size_episodic))
    S′ = reshape(hcat(X[4, :]...), (p.state_size, p.batch_length, p.batch_size_episodic))
    return (S, A, R, S′)
end #sampleBuffer


function sampleBuffer(m::WorldBuffer)
    # @show 𝒟
    # @show size(𝒟) p.episode_length p.batch_size_episodic
    # @show p.batch_length

    start_Points = 1 .+ vcat(0, cumsum(p.world_episode_length)[1:end-1])
    end_Points = cumsum(p.world_episode_length)

    # @show start_Points end_Points

    transInds = vcat([collect(el[1]:(el[2]-p.batch_length)+1) for el in zip(start_Points, end_Points)]...)
    indStart = sample(transInds, p.batch_size_episodic) # to set up dynode_batch_size -> 64 in the paper
    slices = [collect(i:i+p.batch_length-1) for i in indStart]

    # the parameter to loop is S[:,:,i], this returns a matrix of an episode
    batch = 𝒟_World[vcat(slices...)]
    X = hcat(batch...)
    S = reshape(hcat(X[1, :]...), (p.state_size, p.batch_length, p.batch_size_episodic)) #reshape(hcat(X[1,:]...), (2, 40, 128))
    A = reshape(hcat(X[2, :]...), (p.action_size, p.batch_length, p.batch_size_episodic))
    R = reshape(hcat(X[3, :]...), (1, p.batch_length, p.batch_size_episodic))
    S′ = reshape(hcat(X[4, :]...), (p.state_size, p.batch_length, p.batch_size_episodic))
    return (S, A, R, S′)
end #sampleBuffer






# function sampleBuffer(m::MPCBuffer)

#     joint = vcat(𝒟, 𝒟_RL)
#     numEps = size(joint)[1] ÷ p.max_episodes_length #fails when max epiosde length exceeds the size
#     transInds = vcat([collect(((i-1)*p.max_episodes_length+1:i*p.max_episodes_length-p.batch_length)) for i in 1:Int(numEps)]...)
#     indStart = sample(transInds, p.batch_size) # to set up dynode_batch_size -> 64 in the paper
#     slices = [collect(i:i+p.batch_length-1) for i in indStart]

#     # the parameter to loop is S[:,:,i], this returns a matrix of an episode
#     batch = joint[vcat(slices...)]
#     X = hcat(batch...)
#     S = reshape(hcat(X[1, :]...), (p.state_size, p.batch_length, p.batch_size)) #reshape(hcat(X[1,:]...), (2, 40, 128))
#     A = reshape(hcat(X[2, :]...), (p.action_size, p.batch_length, p.batch_size))
#     R = reshape(hcat(X[3, :]...), (1, p.batch_length, p.batch_size))
#     S′ = reshape(hcat(X[4, :]...), (p.state_size, p.batch_length, p.batch_size))
#     return (S, A, R, S′)
# end #sampleBuffer



