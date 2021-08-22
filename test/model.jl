@testset "FourierNeuralOperator" begin
    m = FourierNeuralOperator()

    𝐱, 𝐲 = rand(Float32, 2, 1024, 5), rand(Float32, 1024, 5)
    @test size(m(𝐱)) == size(𝐲)

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    data = [(𝐱[:, :, 1:5], 𝐲[:, 1:5])]
    Flux.train!(loss, params(m), data, Flux.ADAM())
end
