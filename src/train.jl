
function train(algorithm::DDPG, l::Learner)

    scores = zeros(100)
    e = 1
    idx = 1

    while e <= p.max_episodes

        ep = Episode(env, l, p)()


        for (s, a, r, s′, t) in ep.episode

            remember(p.mem_size, s, a, r, s′, t)
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
        

        scores[idx] = ep.total_reward
        idx = idx % 100 + 1
        avg = mean(scores)
        if (e-1) % 10 == 0
            println("Episode: $e | Score: $(ep.total_reward) | Avg score: $avg | Frames: $(p.frames)")
        end
        e += 1

        append!(p.total_rewards, ep.total_reward)

    end

end






function train(algorithm::NODEModel, l::Learner)

    # scores = zeros(100)
    # e = 1
    # idx = 1

    for j in 1:p.Sequences

        ep = Episode(env, l, p)()


        for (s, a, r, s′, t) in ep.episode

            remember(p.mem_size, s, a, r, s′, t)

        end
        

        S, A, R, S′ = sampleBuffer(l.serial)


        for i in 1:p.batch_size
        
            dθ = gradient(() -> loss(NODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(fθ))
            update!(Optimise.Adam(p.τ_actor), Flux.params(fθ), dθ)

            dϕ = gradient(() -> loss(Reward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(Rϕ))
            update!(Optimise.Adam(p.τ_critic), Flux.params(Rϕ), dϕ)

            append!(p.model_loss, loss(NODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
            append!(p.reward_loss, loss(Reward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
        
        end


        if j % 10 == 0
            println("Iteration $j || Model loss $(p.model_loss[end]) || Reward loss $(p.reward_loss[end])")
        end


    end

end



function train(algorithm::DynaWorldModel, l::Learner)


    for j in 1:p.Sequences

        ep = Episode(env, l, p)()


        for (s, a, r, s′, t) in ep.episode

            remember(p.mem_size, s, a, r, s′, t)

        end
        

        S, A, R, S′ = sampleBuffer(l.serial)


        for i in 1:p.batch_size
        
            dθ = gradient(() -> loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(fθ))
            update!(Optimise.Adam(0.005), Flux.params(fθ), dθ)

            dϕ = gradient(() -> loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]), params(Rϕ))
            update!(Optimise.Adam(0.005), Flux.params(Rϕ), dϕ)

            append!(p.model_loss, loss(DyNODE(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
            append!(p.reward_loss, loss(DyReward(), S[:,:,i], A[:,:,i], R[:,:,i], S′[:,:,i]))
        
        end


        if j % 10 == 0
            println("Iteration $j || Model loss $(p.model_loss[end]) || Reward loss $(p.reward_loss[end])")
        end


    end

end






