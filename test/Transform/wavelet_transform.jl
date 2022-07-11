@testset "wavelet transform" begin
    𝐱 = rand(30, 40, 50, 6, 7) # where ch == 6 and batch == 7

    wt = WaveletTransform((3, 4, 5))

    @test size(transform(wt, 𝐱)) == (30, 40, 50, 6, 7)
    @test size(truncate_modes(wt, transform(wt, 𝐱))) == (3, 4, 5, 6, 7)
    @test size(inverse(wt, truncate_modes(wt, transform(wt, 𝐱)))) == (3, 4, 5, 6, 7)
end

@testset "MWT_CZ" begin
    T = Float32
    k = 3
    batch_size = 32

    @testset "MWT_CZ1d" begin
        mwt = MWT_CZ1d()

        # base functions
        wavelet_transform(mwt)
        even_odd(mwt)

        # forward
        Y = mwt(X)

        # backward
        g = gradient()
    end
end
