using NeuralOperators
using Test

@testset "NeuralOperators.jl" begin
    include("preprocess.jl")
    include("fourier.jl")
end
