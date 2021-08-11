using Flux
using NeuralOperators
using Test

@testset "NeuralOperators.jl" begin
    include("data.jl")
    include("fourier.jl")
    include("models.jl")
end
