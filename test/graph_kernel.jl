@testset "GraphKernel" begin
    batch_size = 5
    channel = 32
    coord_dim = 2
    N = 10

    graph = grid([N, N])
    κ = Dense(2(coord_dim + 1), abs2(channel), relu)

    𝐱 = rand(Float32, channel, nv(graph), batch_size)
    E = rand(Float32, 2(coord_dim + 1), ne(graph), batch_size)
    l = WithGraph(FeaturedGraph(graph), GraphKernel(κ, channel))
    @test repr(l.layer) ==
          "GraphKernel(Dense($(2(coord_dim + 1)) => $(abs2(channel)), relu), channel=32)"
    @test size(l(𝐱, E)) == (channel, nv(graph), batch_size)

    g = Zygote.gradient(() -> sum(l(𝐱, E)), Flux.params(l))
    @test length(g.grads) == 3
end
