using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("data.jl")
    include("fourier.jl")
end
