@testitem "NOMAD" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        setups = [
            (u_size=(1, 5), y_size=(1, 5), out_size=(1, 5),
                approximator=(1, 16, 16, 15), decoder=(16, 8, 4, 1), name="Scalar"),
            (u_size=(8, 5), y_size=(2, 5), out_size=(8, 5),
                approximator=(8, 32, 32, 16), decoder=(18, 16, 8, 8), name="Vector")
        ]

        @testset "$(setup.name)" for setup in setups
            u = rand(Float32, setup.u_size...) |> aType
            y = rand(Float32, setup.y_size...) |> aType
            nomad = NOMAD(; approximator=setup.approximator, decoder=setup.decoder)

            ps, st = Lux.setup(rng, nomad) |> dev
            @inferred first(nomad((u, y), ps, st))
            @jet first(nomad((u, y), ps, st))

            pred = first(nomad((u, y), ps, st))
            @test setup.out_size == size(pred)

            __f = (u, y, ps) -> sum(abs2, first(nomad((u, y), ps, st)))
            @test_gradients(__f, u, y, ps; atol=1.0f-3, rtol=1.0f-3)
        end
    end
end
