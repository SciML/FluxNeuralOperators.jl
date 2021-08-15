@testset "SpectralConv" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        SpectralConv(ch, modes)
    )
    @test ndims(SpectralConv(ch, modes)) == 1

    𝐱, _ = get_burgers_data(n=1000)
    @test size(m(𝐱)) == (64, 1024, 1000)

    T = Float32
    loss(x, y) = Flux.mse(m(x), y)
    data = [(T.(𝐱[:, :, 1:5]), rand(T, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end

@testset "FourierOperator" begin
    modes = (16, )
    ch = 64 => 64

    m = Chain(
        Dense(2, 64),
        FourierOperator(ch, modes)
    )

    𝐱, _ = get_burgers_data(n=1000)
    @test size(m(𝐱)) == (64, 1024, 1000)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(Float32.(𝐱[:, :, 1:5]), rand(Float32, 64, 1024, 5))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end
