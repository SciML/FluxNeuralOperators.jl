using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("fourier.jl")
    include("markov.jl")
    include("model.jl")
end
