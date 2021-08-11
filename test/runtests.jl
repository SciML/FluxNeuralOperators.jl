using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("preprocess.jl")
    include("fourier.jl")
end
