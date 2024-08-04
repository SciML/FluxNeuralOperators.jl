@testitem "SpectralConv & SpectralKernel" setup=[SharedTestSetup] begin
    @testset "BACKEND: $(mode)" for (mode, aType, dev, ongpu) in MODES
        rng = StableRNG(12345)

        opconv = [SpectralConv, SpectralKernel]
        setups = [
            (; m=(16,), permuted=Val(false), x_size=(2, 1024, 5), y_size=(128, 1024, 5)),
            (; m=(16,), permuted=Val(true), x_size=(1024, 2, 5), y_size=(1024, 128, 5)),
            (; m=(10, 10), permuted=Val(false),
                x_size=(1, 22, 22, 5), y_size=(64, 22, 22, 5)),
            (; m=(10, 10), permuted=Val(true),
                x_size=(22, 22, 1, 5), y_size=(22, 22, 64, 5))]

        @testset "$(op) $(length(setup.m))D: permuted = $(setup.permuted)" for setup in setups,
            op in opconv

            p = Lux.__unwrap_val(setup.permuted)
            in_chs = ifelse(p, setup.x_size[end - 1], first(setup.x_size))
            out_chs = ifelse(p, setup.y_size[end - 1], first(setup.y_size))
            ch = 64 => out_chs

            l1 = p ? Conv(ntuple(_ -> 1, length(setup.m)), in_chs => first(ch)) :
                 Dense(in_chs => first(ch))
            m = Chain(l1, op(ch, setup.m; setup.permuted))
            display(m)
            ps, st = Lux.setup(rng, m) |> dev

            x = rand(rng, Float32, setup.x_size...) |> aType
            @test size(first(m(x, ps, st))) == setup.y_size
            @inferred m(x, ps, st)
            @jet m(x, ps, st)

            data = [(x, aType(rand(rng, Float32, setup.y_size...)))]
            broken = mode == "AMDGPU"
            @test begin
                l2, l1 = train!(m, ps, st, data; epochs=10)
                l2 < l1
            end broken=broken

            __f = (x, ps) -> sum(abs2, first(m(x, ps, st)))
            test_gradients(__f, x, ps; atol=1f-3, rtol=1f-3)
        end
    end
end
