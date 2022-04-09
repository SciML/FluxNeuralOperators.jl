@testset "1D OperatorConv" begin
    modes = (16, )
    ch = 64 => 128

    m = Chain(
        Dense(2, 64),
        OperatorConv(ch, modes, FourierTransform)
    )
    @test ndims(OperatorConv(ch, modes, FourierTransform)) == 1
    @test repr(OperatorConv(ch, modes, FourierTransform)) == "OperatorConv(64 => 128, (16,), FourierTransform, permuted=false)"

    𝐱 = rand(Float32, 2, 1024, 5)
    @test size(m(𝐱)) == (128, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "permuted 1D OperatorConv" begin
    modes = (16, )
    ch = 64 => 128

    m = Chain(
        Conv((1, ), 2=>64),
        OperatorConv(ch, modes, FourierTransform, permuted=true)
    )
    @test ndims(OperatorConv(ch, modes, FourierTransform, permuted=true)) == 1
    @test repr(OperatorConv(ch, modes, FourierTransform, permuted=true)) == "OperatorConv(64 => 128, (16,), FourierTransform, permuted=true)"

    𝐱 = rand(Float32, 2, 1024, 5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 128, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 128, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "1D OperatorKernel" begin
    modes = (16, )
    ch = 64 => 128

    m = Chain(
        Dense(2, 64),
        OperatorKernel(ch, modes, FourierTransform)
    )
    @test repr(OperatorKernel(ch, modes, FourierTransform)) == "OperatorKernel(64 => 128, (16,), FourierTransform, σ=identity, permuted=false)"

    𝐱 = rand(Float32, 2, 1024, 5)
    @test size(m(𝐱)) == (128, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "permuted 1D OperatorKernel" begin
    modes = (16, )
    ch = 64 => 128

    m = Chain(
        Conv((1, ), 2=>64),
        OperatorKernel(ch, modes, FourierTransform, permuted=true)
    )
    @test repr(OperatorKernel(ch, modes, FourierTransform, permuted=true)) == "OperatorKernel(64 => 128, (16,), FourierTransform, σ=identity, permuted=true)"

    𝐱 = rand(Float32, 2, 1024, 5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 128, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 128, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "2D OperatorConv" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Dense(1, 64),
        OperatorConv(ch, modes, FourierTransform)
    )
    @test ndims(OperatorConv(ch, modes, FourierTransform)) == 2

    𝐱 = rand(Float32, 1, 22, 22, 5)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "permuted 2D OperatorConv" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Conv((1, 1), 1=>64),
        OperatorConv(ch, modes, FourierTransform, permuted=true)
    )
    @test ndims(OperatorConv(ch, modes, FourierTransform, permuted=true)) == 2

    𝐱 = rand(Float32, 1, 22, 22, 5)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "2D OperatorKernel" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Dense(1, 64),
        OperatorKernel(ch, modes, FourierTransform)
    )

    𝐱 = rand(Float32, 1, 22, 22, 5)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "permuted 2D OperatorKernel" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Conv((1, 1), 1=>64),
        OperatorKernel(ch, modes, FourierTransform, permuted=true)
    )

    𝐱 = rand(Float32, 1, 22, 22, 5)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "SpectralConv" begin
    modes = (16, 16)
    ch = 64 => 64
    permuted = false

    @test SpectralConv(ch, modes) isa OperatorConv
    @test SpectralConv(ch, modes).transform isa FourierTransform
end

@testset "GraphKernel" begin
    batch_size = 5
    channel = 32
    N = 10*10

    κ = Dense(2*channel, channel, relu)

    graph = grid([10, 10])
    𝐱 = rand(Float32, channel, N, batch_size)
    l = WithGraph(FeaturedGraph(graph), GraphKernel(κ, channel))
    @test repr(l.layer) == "GraphKernel(Dense(64 => 32, relu), channel=32)"
    @test size(l(𝐱)) == (channel, N, batch_size)

    g = Zygote.gradient(() -> sum(l(𝐱)), Flux.params(l))
    @test length(g.grads) == 3
end
