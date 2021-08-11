using Flux
using NeuralOperators
using Test

tests = [
    "data",
    "fourier",
    "models",
]

@testset "NeuralOperators.jl" begin
    for f in tests
        include("$f.jl")
    end
end
