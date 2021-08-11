@testset "SpectralConv1d" begin
    modes = 16
    ch = 64 => 64

    m = Chain(
        Dense(2, 64, init=NeuralOperators.c_glorot_uniform),
        SpectralConv1d(ch, modes)
    )

    𝐱, _ = get_burgers_data()
    @test size(m(𝐱)) == (64, 1024, 1000)

    T = Float32
    loss(x, y) = Flux.mse(real.(m(x)), y)
    data = [(T.(𝐱[:, :, 1:5]), rand(T, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
    @test true
end

@testset "FourierOperator" begin
    modes = 16
    ch = 64 => 64

    m = Chain(
        Dense(2, 64, init=NeuralOperators.c_glorot_uniform),
        FourierOperator(ch, modes)
    )

    𝐱, _ = get_burgers_data()
    @test size(m(𝐱)) == (64, 1024, 1000)

    T = Float32
    loss(x, y) = Flux.mse(real.(m(x)), y)
    data = [(T.(𝐱[:, :, 1:5]), rand(T, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
    @test true
end
