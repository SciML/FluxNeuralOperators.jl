module NeuralOperators
using Flux
using FFTW
using Tullio
using CUDA
using CUDAKernels
using KernelAbstractions
using Zygote
using ChainRulesCore
using GeometricFlux
using Statistics

export DeepONet
export NOMAD

include("Transform/Transform.jl")
include("operator_kernel.jl")
include("loss.jl")
include("model.jl")
include("DeepONet.jl")
include("subnets.jl")
include("NOMAD.jl")

end
