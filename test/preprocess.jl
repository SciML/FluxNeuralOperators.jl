@testset "get data" begin
    xs, ys = get_data()

    @test size(xs) == (1024, 2, 1000)
    @test size(ys) == (1024, 1000)
end
