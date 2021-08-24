@testset "double pendulum" begin
    xs = DoublePendulum.get_data(i=0, n=100)

    @test size(xs) == (6, 100)
end
