
## Basic DDPG Training Algorithm

function train(algorithm::DDPG, l::Learner)

    scores = zeros(100)
    e = 1
    idx = 1

    while e <= p.max_episodes

        s::Vector{Float32} = env.reset()
        r::Float64 = 0.0
        a::Vector{Float32} = [0.0] # check action space
        t::Bool = false

        episode_rewards = 0

        #ep = Episode(env, l, p)()

        for i in 1:p.max_episodes_length

            p.env_steps += 1
            p.frames += 1

            a = action(l.action_type, l.train, s, p)
            s′, r, t, _ = env.step(a)
            episode_rewards += r
            #noise.X = a
            #@show noise.X

            remember(RandBuffer(), p.mem_size, s, a, r, s′, t)

            if p.frames >= p.train_start# && π.train

                S, A, R, S′ = sampleBuffer(l.serial)

                A′ = μϕ′(S′)
                V′ = Qθ′(vcat(S′, A′))
                Y = R + p.γ * V′

                # critic
                dθ = gradient(() -> loss(Critic(), Y, S, A), Flux.params(Qθ))
                update!(Optimise.Adam(p.η_critic), Flux.params(Qθ), dθ)
                # actor
                dϕ = gradient(() -> -loss(Actor(), S), Flux.params(μϕ))
                update!(Optimise.Adam(p.η_actor), Flux.params(μϕ), dϕ)


                for (base, target) in zip(Flux.params(Qθ), Flux.params(Qθ′))
                    target .= p.τ_critic * base .+ (1 - p.τ_critic) * target
                end

                for (base, target) in zip(Flux.params(μϕ), Flux.params(μϕ′))
                    target .= p.τ_actor * base .+ (1 - p.τ_actor) * target
                end
            end



            s = s′


            if t
                append!(p.episode_length, i)
                env.close()
                break 
            end
            
            
            # if length(𝒟) >= p.train_start# && π.train

            
        end
        
        append!(p.total_rewards, episode_rewards)
        
        scores[idx] = episode_rewards
        idx = idx % 100 + 1
        avg = mean(scores)
        if e % 10 == 0
            #showReward(algorithm, e, avg, p) # Function to replace below output
            println("Episode: $e | Score: $(round(episode_rewards, digits=2)) | Avg score: $(round(avg, digits=2)) | Frames: $(p.frames)")
            #println("Episode: $e | Score: $(ep.total_reward) | Avg score: $avg | Frames: $(p.frames)")
        end
        e += 1

    end

end


## Adjusted training Algorithm for the combined Agent
function trainDDPG(algorithm::modelDDPG) 


    S, A, R, S′ = sampleBuffer(algorithm)

    A′ = μϕ′(S′)
    V′ = Qθ′(vcat(S′, A′))
    Y = R + p.γ * V′

    # critic
    dθ = gradient(() -> loss(Critic(), Y, S, A), Flux.params(Qθ))
    update!(Optimise.Adam(p.η_critic), Flux.params(Qθ), dθ)

    # actor
    dϕ = gradient(() -> -loss(Actor(), S), Flux.params(μϕ))
    update!(Optimise.Adam(p.η_actor), Flux.params(μϕ), dϕ)


    for (base, target) in zip(Flux.params(Qθ), Flux.params(Qθ′))
        target .= p.τ_critic * base .+ (1 - p.τ_critic) * target
    end

    for (base, target) in zip(Flux.params(μϕ), Flux.params(μϕ′))
        target .= p.τ_actor * base .+ (1 - p.τ_actor) * target
    end


end





# Algorithm to learn the model 
function train(algorithm::DynaWorldModel, l::Learner)


    for j in 1:p.Sequences

        ep = Episode(env, l, p)()


        for (s, a, r, s′, t) in ep.episode
 
            remember(RandBuffer(), p.mem_size, s, a, r, s′, t)

        end
        

        S, A, R, S′ = sampleBuffer(l.serial)


        for i in 1:p.batch_size_episodic
        
            dθ = gradient(() -> loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(fθ))
            update!(Optimise.Adam(p.η_node), Flux.params(fθ), dθ)

            dϕ = gradient(() -> loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(Rϕ))
            update!(Optimise.Adam(p.η_reward), Flux.params(Rϕ), dϕ)

            append!(p.model_loss, loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
            append!(p.reward_loss, loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
        
        end

        # some validation

        if j % 50 == 0
            S, A, R, S′ = sampleBuffer(l.serial)
            Ŝ = similar(S)
            R̂ = similar(R)

            for i in 1:p.batch_size_episodic
                Ŝ[:,:,i], R̂[:,:,i] = transitionValidation(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i])
            end

            valLoss = 1/p.batch_size_episodic * (1 / p.state_size) * (1 / p.batch_length) * sum(abs.(copy(Ŝ) - S′))
            append!(p.validation_loss, valLoss)

            println("Iteration $j || Model loss $(round(p.model_loss[end], digits=4)) || Reward loss $(round(p.reward_loss[end], digits=4)) || Validation Loss $(round(valLoss, digits=4))")
        end


    end

end






# Model retrain 
function retrain(algorithm::DynaWorldModel, l::Learner)


    for j in 1:p.model_episode_retrain
        
        S, A, R, S′ = sampleBuffer(Episodic())


        for i in 1:p.batch_size_episodic
        
            dθ = gradient(() -> loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(fθ))
            update!(Optimise.Adam(p.η_node), Flux.params(fθ), dθ)

            dϕ = gradient(() -> loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(Rϕ))
            update!(Optimise.Adam(p.η_reward), Flux.params(Rϕ), dϕ)

            append!(p.model_loss, loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
            append!(p.reward_loss, loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
        
        end


        if j % 5 == 0
            println("Retrain Iteration $j || Model loss $(round(p.model_loss[end], digits=4)) || Reward loss $(round(p.reward_loss[end], digits=4))")
        end


    end

end




function trainOnModel(algorithm::DDPG, l::Learner) #

    scores = zeros(100)
    e = 1
    idx = 1


    while e <= p.max_episodes

        epi = getModelEpisode(env, l, p)


        for (s, a, r, s′, t) in epi

            remember(RandBuffer(), p.mem_size, s, a, r, s′, t)
            p.frames += 1

            # if length(𝒟) >= p.train_start# && π.train
            if p.frames >= p.train_start# && π.train

                S, A, R, S′ = sampleBuffer(l.serial)

                A′ = μϕ′(S′)
                V′ = Qθ′(vcat(S′, A′))
                Y = R + p.γ * V′

                # critic
                dθ = gradient(() -> loss(Critic(), Y, S, A), Flux.params(Qθ))
                update!(Optimise.Adam(p.η_critic), Flux.params(Qθ), dθ)
                # actor
                dϕ = gradient(() -> -loss(Actor(), S), Flux.params(μϕ))
                update!(Optimise.Adam(p.η_actor), Flux.params(μϕ), dϕ)


                for (base, target) in zip(Flux.params(Qθ), Flux.params(Qθ′))
                    target .= p.τ_critic * base .+ (1 - p.τ_critic) * target
                end

                for (base, target) in zip(Flux.params(μϕ), Flux.params(μϕ′))
                    target .= p.τ_actor * base .+ (1 - p.τ_actor) * target
                end
            end
            
        end

        ep = Episode(env, l, p)()

        for (s, a, r, s′, t) in ep.episode
            remember(RandBuffer(), p.mem_size, s, a, r, s′, t)
            p.env_steps += 1

            if p.env_steps % 5 == 0

                S, A, R, S′ = sampleBuffer(l.serial)

                A′ = μϕ′(S′)
                V′ = Qθ′(vcat(S′, A′))
                Y = R + p.γ * V′

                # critic
                dθ = gradient(() -> loss(Critic(), Y, S, A), Flux.params(Qθ))
                update!(Optimise.Adam(p.η_critic), Flux.params(Qθ), dθ)
                # actor
                dϕ = gradient(() -> -loss(Actor(), S), Flux.params(μϕ))
                update!(Optimise.Adam(p.η_actor), Flux.params(μϕ), dϕ)


                for (base, target) in zip(Flux.params(Qθ), Flux.params(Qθ′))
                    target .= p.τ_critic * base .+ (1 - p.τ_critic) * target
                end

                for (base, target) in zip(Flux.params(μϕ), Flux.params(μϕ′))
                    target .= p.τ_actor * base .+ (1 - p.τ_actor) * target
                end
            end
            
        end

        
        
        scores[idx] = ep.total_reward
        idx = idx % 100 + 1
        avg = mean(scores)
        if (e-1) % 25 == 0
            #showReward(algorithm, e, avg, p) # Function to replace below output
            println("Episode: $e | Score: $(round(ep.total_reward, digits=2)) | Avg score: $(round(avg, digits=2)) | Frames: $(p.frames)")
        end
        e += 1
        
        append!(p.total_rewards, ep.total_reward)
        
        if e % 10 == 0 
            if l.model_retrain
                retrain(DynaWorldModel(), l)
            end
        end

    end



end

