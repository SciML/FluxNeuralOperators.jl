using NeuralOperators
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "NeuralOperators.jl" begin
    include("preprocess.jl")
    # include("fourier.jl")
end
