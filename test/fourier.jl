using Flux

@testset "fourier" begin
    modes = 16
    ch = 64 => 64

    m = Chain(
        Conv((1, ), 2=>64),
        SpectralConv1d(ch, modes)
    )

    𝐱, _ = get_data()
    @test size(m(𝐱)) == (1024, 64, 1000)
end

@testset "FNO" begin
    𝐱, 𝐲 = get_data()
    𝐱, 𝐲 = Float32.(𝐱), Float32.(𝐲)
    @test size(FNO()(𝐱)) == size(𝐲)

    # m = FNO()
    # loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    # data = [(𝐱[:, :, 1:5], 𝐲[:, 1:5])]
    # Flux.train!(loss, params(m), data, Flux.ADAM())
end
