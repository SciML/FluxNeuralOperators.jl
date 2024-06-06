using LuxNeuralOperators

include("test_utils.jl")
@testset "DeepONet: $(mode)" for (mode, aType, dev, ongpu) in MODES
    rng_ = get_default_rng(mode)

    u = rand(Float32, 64, 5); # sensor_points x nb
    y = rand(Float32, 1, 10, 5); # ndims x N x nb
    out = rand(Float32, 10, 5);

    don_ = DeepONet(;
        branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16)
    )

    ps, st = Lux.setup(rng_, don_);
    pred = first(don_((u,y),ps,st))
    @test size(pred) == size(out);


    don_ = DeepONet(
        Chain(Dense(64 =>32), Dense(32 =>32), Dense(32 => 16)),
        Chain(Dense(1 =>8), Dense(8 =>8), Dense(8 => 16)),
    )

    ps, st = Lux.setup(rng_, don_);
    pred = first(don_((u,y),ps,st))
    @test size(pred) == size(out);

    don_ =  DeepONet(
        Chain(Dense(64 =>32), Dense(32 =>32), Dense(32 => 20)),
        Chain(Dense(1 =>8), Dense(8 =>8), Dense(8 => 16)),
    )
    ps, st = Lux.setup(rng_, don_);
    @test_throws AssertionError don_((u,y),ps,st)

end