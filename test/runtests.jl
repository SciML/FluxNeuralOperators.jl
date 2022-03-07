using NeuralOperators
using Test
using Flux

@testset "NeuralOperators.jl" begin
    include("Transform/Transform.jl")
    include("operator_kernel.jl")
    include("model.jl")
    include("deeponet.jl")
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
