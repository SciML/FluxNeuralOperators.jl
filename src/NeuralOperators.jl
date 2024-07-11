module NeuralOperators

using ArgCheck: @argcheck
using ChainRulesCore: ChainRulesCore, NoTangent
using ConcreteStructs: @concrete
using FFTW: FFTW, irfft, rfft
using Lux
using LuxCore: LuxCore, AbstractExplicitLayer
using LuxDeviceUtils: get_device, LuxAMDGPUDevice
using NNlib: NNlib, ⊠
using Random: Random, AbstractRNG
using Reexport: @reexport

const CRC = ChainRulesCore

@reexport using Lux

include("utils.jl")
include("transform.jl")

include("functional.jl")
include("layers.jl")

include("fno.jl")
include("deeponet.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export FourierNeuralOperator
export DeepONet

end
