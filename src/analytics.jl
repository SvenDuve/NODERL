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
    UnicodePlots.savefig(diag, "/output/output.txt")


end

function replPlots(t::T, file, p) where T <: Union{DDPG, TW3, SAC}
    
    diag = lineplot(collect(1:length(p.total_rewards)), 
                    Array{Float64}(p.total_rewards),
                    title="Rewards")
    display(diag)
    UnicodePlots.savefig(diag, "/output/output.txt")

end