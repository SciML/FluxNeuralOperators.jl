module NeuralOperators

using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FFTW: FFTW, irfft, rfft
using Random: Random, AbstractRNG
using Static: StaticBool, False, True, known, static, dynamic

using Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer, AbstractLuxWrapperLayer
using LuxLib: batched_matmul
using NNlib: NNlib, batched_adjoint

const CRC = ChainRulesCore

const BoolLike = Union{Bool, StaticBool, Val{true}, Val{false}}

include("utils.jl")
include("transform.jl")

include("functional.jl")
include("layers.jl")

include("fno.jl")
include("deeponet.jl")
include("nomad.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export FourierNeuralOperator
export DeepONet
export NOMAD

end
