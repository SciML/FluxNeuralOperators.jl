@testset "get burgers data" begin
    xs, ys = get_burgers_data(n=1000)

    @test size(xs) == (2, 1024, 1000)
    @test size(ys) == (1024, 1000)
end

@testset "unit gaussian normalizer" begin
    dims = (3, 3, 5, 6)
    𝐱 = rand(Float32, dims)

    n = UnitGaussianNormalizer(𝐱)

    @test size(n.mean) == size(n.std)
    @test size(encode(n, 𝐱)) == dims
    @test size(decode(n, encode(n, 𝐱))) == dims
end

@testset "get darcy flow data" begin
    xs, ys, _, _ = get_darcy_flow_data()

    @test size(xs) == (1, 85, 85, 1024)
    @test size(ys) == (1, 85, 85, 1024)
end
