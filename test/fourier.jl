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
    @test size(FNO()(𝐱)) == size(𝐲)
end
