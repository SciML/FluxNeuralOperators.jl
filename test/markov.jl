@testset "Markov" begin
    m = Chain(
        Dense(1, 10, relu),
        Dense(10, 1, relu),
    )

    mo = MarkovOperator(m, 5)
    @test size(mo(rand(1, 5, 10))) == (1, 5, 10)

    loss(x, y) = Flux.mse(mo(x), y)
    data = [(rand(1, 5, 10), rand(1, 5, 10))]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end
