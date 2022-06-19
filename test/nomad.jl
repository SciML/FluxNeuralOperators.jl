@testset "NOMAD" begin
    @testset "proper construction" begin
        nomad = NOMAD((32,64,72), (24,48,72), σ, tanh)
        # approximator net
        @test size(nomad.approximator_net.layers[end].weight) == (72,64)
        @test size(nomad.approximator_net.layers[end].bias) == (72,)
        # decoder net
        @test size(nomad.decoder_net.layers[end].weight) == (72,48)
        @test size(nomad.decoder_net.layers[end].bias) == (72,)
    end

    # Accept only Int as architecture parameters
    @test_throws MethodError NOMAD((32.5,64,72), (24,48,72), σ, tanh)
    @test_throws MethodError NOMAD((32,64,72), (24.1,48,72))

    # Just the first 16 datapoints from the Burgers' equation dataset
    a = [0.83541104, 0.83479851, 0.83404712, 0.83315711, 0.83212979, 0.83096755,
         0.82967374, 0.82825263, 0.82670928, 0.82504949, 0.82327962, 0.82140651,
         0.81943734, 0.81737952, 0.8152405, 0.81302771]
    sensors = collect(range(0, 1, length=16)')
    model = NOMAD((length(a), 22, length(a)), (length(a) + length(sensors), length(sensors)), σ, tanh; init_approximator=Flux.glorot_normal, bias_decoder=false)
    y = model(a, sensors)
    @test size(y) == (1, 16)
    # Check if model description is printed, when defined
    @test repr(model) == "NOMAD with\nApproximator net: (Chain(Dense(16 => 22, σ), Dense(22 => 16, σ)))\nDecoder net: (Chain(Dense(32 => 16, tanh; bias=false)))\n"

    mgrad = Flux.Zygote.gradient(() -> sum(model(a, sensors)), Flux.params(model))
    @info mgrad.grads
    @test length(mgrad.grads) == 5
end
