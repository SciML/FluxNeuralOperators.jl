using SafeTestsets, Test

@testset "NeuralOperators.jl" begin
    @safetestset "Quality Assurance" include("qa.jl")
    # kernels
    @safetestset "Fourier Transform" include("fourier_transform.jl")
    @safetestset "Chebyshev Transform" include("chebyshev_transform.jl")
    @safetestset "operator_kernel.jl" include("operator_kernel.jl")
    @safetestset "graph kernel" include("graph_kernel.jl")
    @safetestset "loss" include("loss.jl")

    # models
    @safetestset "FNO" include("FNO/FNO.jl")
    @safetestset "DeepONet" include("DeepONet/DeepONet.jl")
    @safetestset "NOMAD" include("NOMAD/NOMAD.jl")
end

#=
    　 ／l、    meow?
    ﾞ（ﾟ､ ｡ ７
    　l、ﾞ ~ヽ
    　じしf_, )ノ
=#
