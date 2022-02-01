using NeuralOperators
using CUDA
using Flux
using GeometricFlux
using Graphs
using LinearAlgebra
using Polynomials
using Zygote
using Test

CUDA.allowscalar(false)

@testset "NeuralOperators.jl" begin
    # kernels
    include("Transform/Transform.jl")
    include("operator_kernel.jl")
    include("graph_kernel.jl")
    include("loss.jl")

    # models
    include("FNO/FNO.jl")
    include("DeepONet/DeepONet.jl")
    include("NOMAD/NOMAD.jl")
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
