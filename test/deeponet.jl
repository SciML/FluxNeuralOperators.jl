@testset "DeepONet" begin
    @testset "proper construction" begin
        deeponet = DeepONet((32,64,72), (24,48,72), σ, tanh)
        # Branch net
        @test size(deeponet.branch_net.layers[end].weight) == (72,64)
        @test size(deeponet.branch_net.layers[end].bias) == (72,)
        # Trunk net
        @test size(deeponet.trunk_net.layers[end].weight) == (72,48)
        @test size(deeponet.trunk_net.layers[end].bias) == (72,)
    end

    # Accept only Int as architecture parameters
    @test_throws MethodError DeepONet((32.5,64,72), (24,48,72), σ, tanh)
    @test_throws MethodError DeepONet((32,64,72), (24.1,48,72))

    # Just the first 16 datapoints from the Burgers' equation dataset
    a = [0.83541104, 0.83479851, 0.83404712, 0.83315711, 0.83212979, 0.83096755,
         0.82967374, 0.82825263, 0.82670928, 0.82504949, 0.82327962, 0.82140651,
         0.81943734, 0.81737952, 0.8152405, 0.81302771]
    sensors = collect(range(0, 1, length=16)')
    model = DeepONet((16, 22, 30), (1, 16, 24, 30), σ, tanh; init_branch=Flux.glorot_normal, bias_trunk=false)
    y = model(a, sensors)
    @test size(y) == (1, 16)

    mgrad = Flux.Zygote.gradient(() -> sum(model(a, sensors)), Flux.params(model))
    @test length(mgrad.grads) == 7

    # Output size of branch and trunk subnets should be same
    branch = Chain(Dense(16, 22), Dense(22, 30))
    trunk = Chain(Dense(1, 16), Dense(16, 24), Dense(24, 32))
    m = DeepONet(branch, trunk)
    @test_throws AssertionError DeepONet((32,64,70), (24,48,72), σ, tanh)
    @test_throws DimensionMismatch m(a, sensors)
end
