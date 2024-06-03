using SafeTestsets, Test

@testset "LuxNeuralOperators.jl" begin
    @safetestset "Layers" begin
        include("layers.jl")
    end
    @safetestset "FNO" begin
        include("fno.jl")
    end
end
