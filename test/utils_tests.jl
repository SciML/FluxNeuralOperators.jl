@testitem "utils" setup=[SharedTestSetup] begin
    import NeuralOperators: deeponet_project, nomad_concatenate, batch_vectorize

    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        setups = [
            (b_size=(16, 5), t_size=(16, 10, 5), out_size=(10, 5),
                additional=NoOpLayer(), name="Scalar"),
            (b_size=(16, 1, 5), t_size=(16, 10, 5), out_size=(1, 10, 5),
                additional=NoOpLayer(), name="Scalar II"),
            (b_size=(16, 3, 5), t_size=(16, 10, 5), out_size=(3, 10, 5),
                additional=NoOpLayer(), name="Vector"),
            (b_size=(16, 4, 3, 3, 5), t_size=(16, 10, 5),
                out_size=(4, 3, 3, 10, 5), additional=NoOpLayer(), name="Tensor"),
            (b_size=(16, 5), t_size=(16, 10, 5), out_size=(4, 10, 5),
                additional=Dense(16 => 4), name="additional : Scalar"),
            (b_size=(16, 1, 5), t_size=(16, 10, 5), out_size=(4, 1, 10, 5),
                additional=Dense(16 => 4), name="additional : Scalar II"),
            (b_size=(16, 3, 5), t_size=(16, 10, 5), out_size=(4, 3, 10, 5),
                additional=Dense(16 => 4), name="additional : Vector"),
            (b_size=(16, 4, 3, 3, 5), t_size=(16, 10, 5), out_size=(3, 4, 3, 4, 10, 5),
                additional=Chain(Dense(16 => 4), ReshapeLayer((3, 4, 3, 4, 10))),
                name="additional : Tensor")]

        @testset "project : $(setup.name)" for setup in setups
            b = rand(Float32, setup.b_size...) |> aType
            t = rand(Float32, setup.t_size...) |> aType

            ps, st = Lux.setup(rng, setup.additional) |> dev
            additional = setup.additional isa NoOpLayer ? nothing :
                         StatefulLuxLayer{true}(setup.additional, ps, st)

            @inferred deeponet_project(b, t, additional)
            @jet deeponet_project(b, t, additional)
            @test setup.out_size == size(deeponet_project(b, t, additional))
        end

        setups = [(x_size=(6, 5), y_size=(4, 5), out_size=(10, 5), name="Scalar"),
            (x_size=(12, 5), y_size=(8, 5), out_size=(20, 5), name="Vector I"),
            (x_size=(4, 6, 5), y_size=(6, 5), out_size=(30, 5), name="Vector II"),
            (x_size=(4, 2, 3, 5), y_size=(2, 2, 3, 5), out_size=(36, 5), name="Tensor")]

        @testset "nomad_concatenate $(setup.name)" for setup in setups
            x_size = rand(Float32, setup.x_size...) |> aType
            y_size = rand(Float32, setup.y_size...) |> aType

            @test setup.out_size == size(nomad_concatenate(x_size, y_size))
        end

        @testset "batch vectorize" begin
            x_size = (4, 2, 3)
            x = rand(Float32, x_size..., 5) |> aType
            @test size(batch_vectorize(x)) == (prod(x_size), 5)
        end
    end
end
