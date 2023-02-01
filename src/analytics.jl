function showResults(t::T, p) where T <: NODEArchitecture

    plot(p.model_loss, label="Model Loss")
    plot!(twinx(), p.reward_loss, label="Reward Loss")

end


function showResults(t::T, p) where T <: Union{DDPG, TW3, SAC}
    plot(p.total_rewards, label="Total Rewards")
end


function replPlots(t::T, file, p) where T <: NODEArchitecture
    
    # lineplot(collect(1:length(p.model_loss)), 
    #             [Array{Float64}(p.model_loss) Array{Float64}(p.reward_loss)], 
    #             color=[:green :red],
    #             name=["Model", "Reward"])
    diag = lineplot(collect(1:length(p.model_loss)), 
                Array{Float64}(p.model_loss),
                title="Model Loss")
    # lineplot(collect(1:length(p.reward_loss)), 
    #             Array{Float64}(p.reward_loss),
    #             color=:cyan,
    #             title="Reward Loss")
    display(diag)
    UnicodePlots.savefig(diag, "output/" * file * ".txt")


end

function replPlots(t::T, file, p) where T <: Union{DDPG, TW3, SAC}
    
    diag = lineplot(collect(1:length(p.total_rewards)), 
                    Array{Float64}(p.total_rewards),
                    title="Rewards", color=:red)
    display(diag)
    UnicodePlots.savefig(diag, "output/" * file * ".txt")

end


function storePlots(t::T, file, p) where T <: Union{DDPG, TW3, SAC}
    
    diag = plot(collect(1:length(p.total_rewards)), 
                    Array{Float64}(p.total_rewards),
                    title="Rewards", color=:red)
    #display(diag)
    Plots.savefig(diag, "output/" * file * ".png")

end


function storePlots(t::T, file, p) where T <: NODEArchitecture
    
    modl = plot(collect(1:length(p.model_loss)), 
                    Array{Float64}(p.model_loss),
                    title="Model Loss", color=:red)
    Plots.savefig(modl, "output/" * file * "_model" * ".png")
    rew = plot(collect(1:length(p.reward_loss)), 
                    Array{Float64}(p.reward_loss),
                    title="Reward Loss", color=:green)
    Plots.savefig(rew, "output/" * file * "_reward" * ".png")
    #display(diag)

end




function showReward(m::Agent, e, avg, p) 
    println("Episode: $e | Score: $(round(ep.total_reward, digits=2)) | Avg score: $(round(avg, digits=2)) | Frames: $(p.frames)")
end


function showAgent(file, pms::Parameter) 

    global μϕ = loadModel(file, pms)

    gym = pyimport("gym")
    global env = gym.make(pms.environment)
    global p = resetParameters(pms)


    s = env.reset()
    R = []
    notSolved = true

    while notSolved

        a = action(Clamped(), false, s, p) #action(t::Clamped, m::Bool, s::Vector{Float32}, p::Parameter)

        s′, r, t, _ = env.step(a)
        append!(R, r)
        env.render()
        sleep(0.1)
        s = s′
        notSolved = !t
    end

    env.close()

end


function storeModel(policy, file, p) 
    @save file policy
end


function loadModel(file, p) 
    @load file policy
    return policy
end


