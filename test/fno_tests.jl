@testitem "Fourier Neural Operator" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        setups = [
            (modes=(16,), chs=(2, 64, 64, 64, 64, 64, 128, 1),
                x_size=(2, 1024, 5), y_size=(1, 1024, 5), permuted=Val(false)),
            (modes=(16,), chs=(2, 64, 64, 64, 64, 64, 128, 1),
                x_size=(1024, 2, 5), y_size=(1024, 1, 5), permuted=Val(true))]

        @testset "$(length(setup.modes))D: permuted = $(setup.permuted)" for setup in setups
            fno = FourierNeuralOperator(; setup.chs, setup.modes, setup.permuted)
            display(fno)
            ps, st = Lux.setup(rng, fno) |> dev

            x = rand(rng, Float32, setup.x_size...) |> aType
            y = rand(rng, Float32, setup.y_size...) |> aType

            @inferred fno(x, ps, st)
            @jet fno(x, ps, st)

            @test size(first(fno(x, ps, st))) == setup.y_size

            data = [(x, y)]
            broken = mode == "AMDGPU"
            @test begin
                l2, l1 = train!(fno, ps, st, data; epochs=10)
                l2 < l1
            end broken=broken
        end
    end
end
