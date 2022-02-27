@testset "DeepONet Training Accuracy" begin
    ϵ = Burgers.train_don()

    @test ϵ < 0.4
end
