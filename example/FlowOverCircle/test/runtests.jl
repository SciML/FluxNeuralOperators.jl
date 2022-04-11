using FlowOverCircle
using FluxTraining
using Test

@testset "FlowOverCircle" begin
    learner = FlowOverCircle.train(epochs=5)
    loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
    @test loss < 0.1
end
