@testset "get double pendulum chaotic data" begin
    xs = get_double_pendulum_chaotic_data(i=0, n=100)

    @test size(xs) == (6, 100)
end
