@testset "FourierNeuralOperator" begin
    m = FourierNeuralOperator()

    𝐱, 𝐲 = rand(Float32, 2, 1024, 5), rand(Float32, 1, 1024, 5)
    @test size(m(𝐱)) == size(𝐲)

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    data = [(𝐱, 𝐲)]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end

@testset "MarkovNeuralOperator" begin
    m = MarkovNeuralOperator()

    𝐱, 𝐲 = rand(Float32, 1, 64, 64, 5), rand(Float32, 1, 64, 64, 5)
    @test size(𝐱) == size(m(𝐱)) == size(𝐲)

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    data = [(𝐱, 𝐲)]
    Flux.train!(loss, Flux.params(m), data, Flux.ADAM())
end
