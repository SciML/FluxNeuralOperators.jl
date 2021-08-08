@testset "get data" begin
    xs, ys = get_data()
    loc_xs, ys = preprocess(xs, ys)

    @test size(loc_xs) == (1024, 2, 1000)
    @test size(ys) == (1024, 1000)

    @test loc_xs[:, 1, 1] == LinRange(0, 1, size(xs, 1))
    @test loc_xs[:, 2, :] == xs
end
