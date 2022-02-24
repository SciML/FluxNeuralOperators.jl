using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("fourier.jl")
    include("model.jl")
    include("deeponet.jl")
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
