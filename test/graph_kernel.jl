@testset "GraphKernel" begin
    batch_size = 5
    channel = 32
    N = 10 * 10

    κ = Dense(2 * channel, channel, relu)

    graph = grid([10, 10])
    𝐱 = rand(Float32, channel, N, batch_size)
    l = WithGraph(FeaturedGraph(graph), GraphKernel(κ, channel))
    @test repr(l.layer) == "GraphKernel(Dense(64 => 32, relu), channel=32)"
    @test size(l(𝐱)) == (channel, N, batch_size)

    g = Zygote.gradient(() -> sum(l(𝐱)), Flux.params(l))
    @test length(g.grads) == 3
end
