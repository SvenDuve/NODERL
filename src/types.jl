abstract type RL end

abstract type Agent <: RL end
abstract type Model <: RL end

abstract type Process <: RL end
# Concrete  
struct Episodic <: Process end
struct Online <: Process end



abstract type ActionSelection <: RL end
# Concrete
struct Randomized <: ActionSelection end
struct Clamped <: ActionSelection end

# abstract type NN <: RL end
# abstract type Node <: RL end


# Concrete Algorithms
struct DDPG <: Agent end
struct TW3 <: Agent end 
struct SAC <: Agent end

#struct DyNode <: Model end
struct DynaWorldModel <: Model end
struct NNModel <: Model end
struct NODEModel <: Model end


abstract type NetworkArchitecture <: RL end

struct Critic <: NetworkArchitecture end
struct Actor <: NetworkArchitecture end
struct Reward <: NetworkArchitecture end
struct NODE <: NetworkArchitecture end
struct DyNODE <: NetworkArchitecture end
struct DyReward <: NetworkArchitecture end


const NODEArchitecture = Union{NODEModel, DynaWorldModel, NODE}
const TransitionType = Union{DyNODE, DyReward}

abstract type NoiseGenerator <: RL end


mutable struct OrnsteinUhlenbeck <: NoiseGenerator
    μ
    θ
    σ
    X
end

mutable struct GaussianNoise <: NoiseGenerator
    μ
    σ
end



# Concrete  

mutable struct Learner <: RL
    train::Bool
    algorithm::RL
    serial::Process
    action_type::ActionSelection
    function Learner(algorithm::RL, serial::Process, action_type::ActionSelection)
        return new(true, algorithm, serial, action_type)
    end
end
