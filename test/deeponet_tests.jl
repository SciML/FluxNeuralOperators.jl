@testitem "DeepONet" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng_ = get_stable_rng()

        u = rand(64, 5) |> aType # sensor_points x nb
        y = rand(1, 10, 5) |> aType # ndims x N x nb
        out_size = (10, 5)

        don_ = DeepONet(;
            branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16)
        )

        ps, st = Lux.setup(rng_, don_) |> dev
        
        @inferred don_((u,y),ps,st)
        @jet don_((u,y),ps,st)

        pred = first(don_((u,y),ps,st))
        @test size(pred) == out_size

        don_ = DeepONet(
            Chain(Dense(64 =>32), Dense(32 =>32), Dense(32 => 16)),
            Chain(Dense(1 =>8), Dense(8 =>8), Dense(8 => 16)),
        )

        ps, st = Lux.setup(rng_, don_) |> dev

        @inferred don_((u,y),ps,st)
        @jet don_((u,y),ps,st)

        pred = first(don_((u,y),ps,st))
        @test size(pred) == out_size

        don_ =  DeepONet(
            Chain(Dense(64 =>32), Dense(32 =>32), Dense(32 => 20)),
            Chain(Dense(1 =>8), Dense(8 =>8), Dense(8 => 16)),
        )
        ps, st = Lux.setup(rng_, don_) |> dev
        @test_throws AssertionError don_((u,y),ps,st)
    end

end