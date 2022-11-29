@testset "Fourier transform" begin
    ch = 6
    batch = 7
    𝐱 = rand(30, 40, 50, ch, batch)

    ft = FourierTransform((3, 4, 5))

    @test size(transform(ft, 𝐱)) == (16, 40, 50, ch, batch)
    @test size(truncate_modes(ft, transform(ft, 𝐱))) == (3, 4, 5, ch, batch)
    @test size(inverse(ft,
                       NeuralOperators.pad_modes(truncate_modes(ft, transform(ft, 𝐱)),
                                                 size(transform(ft, 𝐱))),
                       size(𝐱))) == (30, 40, 50, ch, batch)

    g = Zygote.gradient(x -> sum(inverse(ft,
                                         NeuralOperators.pad_modes(truncate_modes(ft,
                                                                                  transform(ft,
                                                                                            x)),
                                                                   (16, 40, 50, ch, batch)),
                                         (30, 40, 50, ch, batch))), 𝐱)
    @test size(g[1]) == (30, 40, 50, ch, batch)
end
