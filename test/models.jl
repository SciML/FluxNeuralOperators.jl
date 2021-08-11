@testset "FNO" begin
    𝐱, 𝐲 = get_burgers_data()
    𝐱, 𝐲 = Float32.(𝐱), Float32.(𝐲)
    @test size(FNO()(𝐱)) == size(𝐲)

    m = FNO()
    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]
    data = [(𝐱[:, :, 1:5], 𝐲[:, 1:5])]
    Flux.train!(loss, params(m), data, Flux.ADAM())
    @test true
end
