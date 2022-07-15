@testset "GraphKernel" begin
    batch_size = 5
    channel = 1
    coord_dim = 2
    N = 10

    graph = grid([N, N])
    𝐱 = rand(Float32, channel, nv(graph), batch_size)
    κ = Dense(2(coord_dim + channel), abs2(channel), relu)
    κ_in_dim, κ_out_dim = 2(coord_dim + channel), abs2(channel)

    @testset "pass edge features" begin
        E = rand(Float32, 2(coord_dim + channel), ne(graph), batch_size)
        l = WithGraph(FeaturedGraph(graph), GraphKernel(κ, channel))
        @test repr(l.layer) ==
              "GraphKernel(Dense($κ_in_dim => $κ_out_dim, relu), channel=$channel)"
        @test size(l(𝐱, E)) == (channel, nv(graph), batch_size)

        g = Zygote.gradient(() -> sum(l(𝐱, E)), Flux.params(l))
        @test length(g.grads) == 3
    end

    @testset "pass positional features" begin
        pf = rand(Float32, coord_dim, nv(graph), batch_size)
        pf = vcat(𝐱, pf)
        fg = FeaturedGraph(graph)
        l = WithGraph(fg, GraphKernel(κ, channel))
        @test repr(l.layer) ==
              "GraphKernel(Dense($κ_in_dim => $κ_out_dim, relu), channel=$channel)"
        @test size(l(pf, 𝐱, nothing)) == (channel, nv(graph), batch_size)

        g = Zygote.gradient(() -> sum(l(pf, 𝐱, nothing)), Flux.params(l))
        @test length(g.grads) == 4
    end

    @testset "pass positional features by FeaturedGraph" begin
        pf = rand(Float32, coord_dim, nv(graph), batch_size)
        pf = vcat(𝐱, pf)
        fg = FeaturedGraph(graph, pf = pf)
        l = WithGraph(fg, GraphKernel(κ, channel))
        @test repr(l.layer) ==
              "GraphKernel(Dense($κ_in_dim => $κ_out_dim, relu), channel=$channel)"
        @test size(l(𝐱)) == (channel, nv(graph), batch_size)

        g = Zygote.gradient(() -> sum(l(𝐱)), Flux.params(l))
        @test length(g.grads) == 3
    end
end
