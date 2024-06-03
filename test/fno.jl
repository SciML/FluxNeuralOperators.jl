using LuxNeuralOperators

include("test_utils.jl")

@testset "FourierNeuralOperator: $(mode)" for (mode, aType, dev, ongpu) in MODES
    rng = get_default_rng(mode)

    setups = [
        (modes=(16,), chs=(2, 64, 64, 64, 64, 64, 128, 1), x_size=(2, 1024, 5),
            y_size=(1, 1024, 5), permuted=Val(false)),
        (modes=(16,), chs=(2, 64, 64, 64, 64, 64, 128, 1), x_size=(1024, 2, 5),
            y_size=(1024, 1, 5), permuted=Val(true))]

    @testset "$(length(setup.modes))D: permuted = $(setup.permuted)" for setup in setups
        fno = FourierNeuralOperator(rng; setup.chs, setup.modes, setup.permuted)

        x = rand(rng, Float32, setup.x_size...)
        y = rand(rng, Float32, setup.y_size...)

        ps, st = Lux.setup(rng, fno)

        @test size(first(fno(x, ps, st))) == setup.y_size

        data = [(x, y)]
        l2, l1 = train!(fno, ps, st, data; epochs=10)
        @test l2 < l1
    end
end
