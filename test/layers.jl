using LuxNeuralOperators

include("test_utils.jl")

@testset "Layers: $(mode)" for (mode, aType, dev, ongpu) in MODES
    rng = get_default_rng(mode)

    opconv = [SpectralConv, SpectralKernel]
    setups = [
        (; m=(16,), permuted=Val(false), x_size=(2, 1024, 5), y_size=(128, 1024, 5)),
        (; m=(16,), permuted=Val(true), x_size=(1024, 2, 5), y_size=(1024, 128, 5)),
        (; m=(10, 10), permuted=Val(false), x_size=(1, 22, 22, 5), y_size=(64, 22, 22, 5)),
        (; m=(10, 10), permuted=Val(true), x_size=(22, 22, 1, 5), y_size=(22, 22, 64, 5))
    ]

    loss(m, ps, x, y) = sum(abs2, m(x, ps) .- y)

    @testset "$(length(setup.m))D $(op): permuted = $(setup.permuted)" for setup in setups,
        op in opconv

        p = Lux.__unwrap_val(setup.permuted)
        in_chs = ifelse(p, setup.x_size[end - 1], first(setup.x_size))
        out_chs = ifelse(p, setup.y_size[end - 1], first(setup.y_size))
        ch = 64 => out_chs
        l1 = p ? Conv(ntuple(_ -> 1, length(setup.m)), in_chs => first(ch)) :
             Dense(in_chs => first(ch))
        m = Chain(l1, op(rng, ch, setup.m; setup.permuted))
        ps, st = Lux.setup(rng, m)

        x = rand(rng, Float32, setup.x_size...)
        @test size(first(m(x, ps, st))) == setup.y_size
        @inferred m(x, ps, st)

        data = [(x, rand(rng, Float32, setup.y_size...))]
        l2, l1 = train!(loss, m, ps, st, data; epochs=10)
        @test l2 < l1
    end
end
