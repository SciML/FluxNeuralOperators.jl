@testset "CUDA" begin
    @testset "DeepONet" begin
        batch_size = 2
        a = [0.83541104, 0.83479851, 0.83404712, 0.83315711, 0.83212979, 0.83096755,
             0.82967374, 0.82825263, 0.82670928, 0.82504949, 0.82327962, 0.82140651,
             0.81943734, 0.81737952, 0.8152405, 0.81302771]
        a = repeat(a, outer=(1, batch_size)) |> gpu
        sensors = collect(range(0, 1, length=16)')
        sensors = repeat(sensors, outer=(batch_size, 1)) |> gpu
        model = DeepONet((16, 22, 30), (2, 16, 24, 30), σ, tanh;
            init_branch=Flux.glorot_normal, bias_trunk=false) |> gpu
        y = model(a, sensors)
        @test size(y) == (batch_size, 16)

        mgrad = Flux.Zygote.gradient(() -> sum(model(a, sensors)), Flux.params(model))
        @test length(mgrad.grads) == 9
    end
end
