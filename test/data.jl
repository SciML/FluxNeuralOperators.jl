@testset "get data" begin
    xs, ys = get_burgers_data()

    @test size(xs) == (2, 1024, 1000)
    @test size(ys) == (1024, 1000)
end
