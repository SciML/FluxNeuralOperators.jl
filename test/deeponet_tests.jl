@testitem "DeepONet" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        u = rand(Float32, 64, 5) |> aType # sensor_points x nb
        y = rand(Float32, 1, 10, 5) |> aType # ndims x N x nb
        out_size = (10, 5)

        deeponet = DeepONet(; branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16))
        display(deeponet)
        ps, st = Lux.setup(rng, deeponet) |> dev

        @inferred deeponet((u, y), ps, st)
        @jet deeponet((u, y), ps, st)

        pred = first(deeponet((u, y), ps, st))
        @test size(pred) == out_size

        deeponet = DeepONet(Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 16)),
            Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16)))
        display(deeponet)
        ps, st = Lux.setup(rng, deeponet) |> dev

        @inferred deeponet((u, y), ps, st)
        @jet deeponet((u, y), ps, st)

        pred = first(deeponet((u, y), ps, st))
        @test size(pred) == out_size

        deeponet = DeepONet(Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 20)),
            Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16)))
        display(deeponet)
        ps, st = Lux.setup(rng, deeponet) |> dev
        @test_throws ArgumentError deeponet((u, y), ps, st)

        @testset "higher-dim input #7" begin
            u = ones(Float32, 10, 10, 10) |> aType
            v = ones(Float32, 1, 10, 10) |> aType
            deeponet = DeepONet(; branch=(10, 10, 10), trunk=(1, 10, 10))
            display(deeponet)
            ps, st = Lux.setup(rng, deeponet) |> dev

            y, st_ = deeponet((u, v), ps, st)
            @test size(y) == (10, 10)

            @inferred deeponet((u, v), ps, st)
            @jet deeponet((u, v), ps, st)
        end
    end
end
