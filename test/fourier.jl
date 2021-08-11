@testset "SpectralConv1d" begin
    modes = 16
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        SpectralConv1d(ch, modes)
    )

    𝐱, _ = get_data()
    @test size(m(𝐱)) == (64, 1024, 1000)

    T = Float32
    loss(x, y) = Flux.mse(m(x), y)
    data = [(T.(𝐱[:, :, 1:5]), rand(T, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperator" begin
    modes = 16
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        FourierOperator(ch, modes)
    )

    𝐱, _ = get_data()
    @test size(m(𝐱)) == (64, 1024, 1000)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(Float32.(𝐱[:, :, 1:5]), rand(Float32, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FNO" begin
    𝐱, 𝐲 = get_data()
    𝐱, 𝐲 = Float32.(𝐱), Float32.(𝐲)
    @test size(FNO()(𝐱)) == size(𝐲)

    m = FNO()
    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    data = [(𝐱[:, :, 1:5], 𝐲[:, 1:5])]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end
