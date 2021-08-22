using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("fourier.jl")
    include("model.jl")
end
