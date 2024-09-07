module NeuralOperators

using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FFTW: FFTW, irfft, rfft
using Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer
using LuxLib: batched_matmul
using NNlib: NNlib, batched_adjoint
using Random: Random, AbstractRNG

const CRC = ChainRulesCore

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
