using NeuralOperators
using Test
using Flux
using CUDA

CUDA.allowscalar(false)

cuda_tests = [
    "cuda",
]

tests = [
    "Transform/Transform",
    "operator_kernel",
    "model",
    "deeponet",
]

if CUDA.functional()
    append!(tests, cuda_tests)
else
    @warn "CUDA unavailable, not testing GPU support"
end

@testset "NeuralOperators.jl" begin
    for t in tests
        include("$(t).jl")
    end
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
