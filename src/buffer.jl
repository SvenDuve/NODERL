

function remember(mem_size, s::Vector{Float32}, a::Vector{Float64}, r::Float64, s′::Vector{Float32}, t::Bool)
    if length(𝒟) >= mem_size
        deleteat!(𝒟, 1)
    end
    push!(𝒟, [s, a, r, s′, t])
end #remember


function sampleBuffer(m::Process)
    minibatch = sample(𝒟, p.batch_size)
    X = hcat(minibatch...)
    S = hcat(X[1, :]...)
    A = hcat(X[2, :]...)
    R = hcat(X[3, :]...)
    S′ = hcat(X[4, :]...)
    return (S, A, R, S′)
end #sampleBuffer



function sampleBuffer(m::Episodic)

    numEps = size(𝒟)[1] ÷ p.max_episodes_length #fails when max epiosde length exceeds the size
    transInds = vcat([collect(((i-1)*p.max_episodes_length+1:i*p.max_episodes_length-p.batch_length)) for i in 1:Int(numEps)]...)
    indStart = sample(transInds, p.batch_size) # to set up dynode_batch_size -> 64 in the paper
    slices = [collect(i:i+p.batch_length-1) for i in indStart]

    # the parameter to loop is S[:,:,i], this returns a matrix of an episode
    batch = 𝒟[vcat(slices...)]
    X = hcat(batch...)
    S = reshape(hcat(X[1, :]...), (p.state_size, p.batch_length, p.batch_size)) #reshape(hcat(X[1,:]...), (2, 40, 128))
    A = reshape(hcat(X[2, :]...), (p.action_size, p.batch_length, p.batch_size))
    R = reshape(hcat(X[3, :]...), (1, p.batch_length, p.batch_size))
    S′ = reshape(hcat(X[4, :]...), (p.state_size, p.batch_length, p.batch_size))
    return (S, A, R, S′)
end #sampleBuffer



