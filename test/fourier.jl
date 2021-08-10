using Flux

@testset "fourier" begin
    modes = 16
    width = 64
    ch = width => width
    m = Chain(
        Conv((1, ), 2=>width),
        SpectralConv1d(ch, modes)
    )

    𝐱, _ = get_data()
    @show size(m(𝐱))
end
