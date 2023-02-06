abstract type RL end

abstract type MBDDPG <: RL end
abstract type Agent <: RL end
abstract type Model <: RL end

abstract type Process <: RL end
# Concrete  
struct Episodic <: Process end
struct Online <: Process end
struct PER <: Process end



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
struct modelDDPG <: Model end


abstract type NetworkArchitecture <: RL end

struct Critic <: NetworkArchitecture end
struct Actor <: NetworkArchitecture end
struct Reward <: NetworkArchitecture end
struct NODE <: NetworkArchitecture end
struct DyNODE <: NetworkArchitecture end
struct DyReward <: NetworkArchitecture end

abstract type GeneralBuffer <: RL end
struct RandBuffer <: GeneralBuffer end
struct WorldBuffer <: GeneralBuffer end

abstract type Noise <: RL end
struct Exponential <: Noise end
struct Adaptive <: Noise end
struct Static <: Noise end

#struct RandBuffer <: GeneralBuffer end


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

mutable struct NoiseFree <: NoiseGenerator end


# Concrete  

mutable struct Learner <: RL
    train::Bool
    model_retrain::Bool
    algorithm::RL
    serial::Process
    action_type::ActionSelection
    function Learner(algorithm::RL, serial::Process, action_type::ActionSelection)
        return new(true, true, algorithm, serial, action_type)
    end
end
