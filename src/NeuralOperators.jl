module NeuralOperators

using Flux
using FFTW
using Tullio
using CUDA
using KernelAbstractions
using Zygote
using ChainRulesCore
using GeometricFlux
using Statistics
using Polynomials
using SpecialPolynomials
using LinearAlgebra

include("abstracttypes.jl")

# kernels
include("Transform/Transform.jl")
include("operator_kernel.jl")
include("graph_kernel.jl")
include("loss.jl")

# models
include("FNO/FNO.jl")
include("DeepONet/DeepONet.jl")
include("NOMAD/NOMAD.jl")

end # module
