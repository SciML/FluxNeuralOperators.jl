@testset "loss" begin
    𝐲 = rand(1, 3, 3, 5)
    𝐲̂ = rand(1, 3, 3, 5)

    feature_dims = 2:3
    loss = mean(.√(sum(abs2, 𝐲̂-𝐲, dims=feature_dims)))

    @test l₂loss(𝐲̂, 𝐲) ≈ loss/prod(feature_dims)
end
