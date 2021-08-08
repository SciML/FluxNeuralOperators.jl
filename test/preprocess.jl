@testset "get data" begin
    xs, ys = get_data()
    @test size(xs) == (1000, 1024, 2)
    @test size(ys) == (1000, 1024)
end
