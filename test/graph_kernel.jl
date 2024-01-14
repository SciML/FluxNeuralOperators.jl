using NeuralOperators
using Flux
using Graphs
using GeometricFlux
batch_size = 5
channel = 1
coord_dim = 2
N = 10

graph = grid([N, N])
𝐱 = rand(Float32, channel, nv(graph), batch_size)
κ = Dense(2(coord_dim + channel), abs2(channel), relu)
κ_in_dim, κ_out_dim = 2(coord_dim + channel), abs2(channel)

@testset "layer without graph" begin
    pf = rand(Float32, coord_dim, nv(graph), batch_size)
    pf = vcat(𝐱, pf)
    l = GraphKernel(κ, channel)
    fg = FeaturedGraph(graph, nf = 𝐱, pf = pf)
    fg_ = l(fg)
    @test size(node_feature(fg_)) == (channel, nv(graph), batch_size)
    @test_throws MethodError l(𝐱)

    g = gradient(() -> sum(node_feature(l(fg))), Flux.params(l))
    @test length(g.grads) == 5
end
