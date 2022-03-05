@testset "fourier transform" begin
    𝐱 = rand(30, 40, 50, 6, 7) # where ch == 6 and batch == 7

    ft = FourierTransform((3, 4, 5))

    @test size(transform(ft, 𝐱)) == (30, 40, 50, 6, 7)
    @test size(truncate_modes(ft, transform(ft, 𝐱))) == (3, 4, 5, 6, 7)
    @test size(inverse(ft, truncate_modes(ft, transform(ft, 𝐱)))) == (3, 4, 5, 6, 7)
end
