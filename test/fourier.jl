@testset "SpectralConv1d" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        SpectralConv(ch, modes)
    )
    @test ndims(SpectralConv(ch, modes)) == 1

    𝐱, _ = get_burgers_data(n=5)
    @test size(m(𝐱)) == (64, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "SpectralConvPerm1d" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Conv((1, ), 2=>64),
        SpectralConv(ch, modes, permuted=true)
    )
    @test ndims(SpectralConv(ch, modes, permuted=true)) == 1

    𝐱, _ = get_burgers_data(n=5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 64, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperator1d" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        FourierOperator(ch, modes)
    )

    𝐱, _ = get_burgers_data(n=5)
    @test size(m(𝐱)) == (64, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperatorPerm1d" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Conv((1, ), 2=>64),
        FourierOperator(ch, modes, permuted=true)
    )

    𝐱, _ = get_burgers_data(n=5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 64, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "SpectralConv2d" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Dense(1, 64),
        SpectralConv(ch, modes)
    )
    @test ndims(SpectralConv(ch, modes)) == 2

    𝐱, _, _, _ = get_darcy_flow_data(n=5, Δsamples=20)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "SpectralConvPerm2d" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Conv((1, 1), 1=>64),
        SpectralConv(ch, modes, permuted=true)
    )
    @test ndims(SpectralConv(ch, modes, permuted=true)) == 2

    𝐱, _, _, _ = get_darcy_flow_data(n=5, Δsamples=20)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperator2d" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Dense(1, 64),
        FourierOperator(ch, modes)
    )

    𝐱, _, _, _ = get_darcy_flow_data(n=5, Δsamples=20)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperatorPerm2d" begin
    modes = (16, 16)
    ch = 64 => 64

    m = Chain(
        Conv((1, 1), 1=>64),
        FourierOperator(ch, modes, permuted=true)
    )

    𝐱, _, _, _ = get_darcy_flow_data(n=5, Δsamples=20)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end
