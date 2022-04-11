using NeuralOperators
using CUDA
using Flux
using GeometricFlux
using Graphs
using Zygote
using Test

CUDA.allowscalar(false)

cuda_tests = [
    "cuda.jl",
]

tests = [
    "Transform/Transform.jl",
    "operator_kernel.jl",
    "loss.jl",
    "model.jl",
    "deeponet.jl",
]

if CUDA.functional()
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "NeuralOperators.jl" begin
    for t in tests
        include(t)
    end
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
