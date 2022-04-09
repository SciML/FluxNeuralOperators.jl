using DoublePendulum
using Test

@testset "DoublePendulum" begin
    xs = DoublePendulum.get_data(i=0, n=100)

    @test size(xs) == (6, 100)

    learner = DoublePendulum.train(epochs=5)
    loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
    @test loss < 0.05
end
