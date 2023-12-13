using Random, StableRNGs, Statistics, Test
using Lux, LuxCUDA, Zygote, Optimisers
using LuxTestUtils: @jet, @test_gradients

CUDA.allowscalar(false)

const GROUP = get(ENV, "GROUP", "All")

cpu_testing() = GROUP == "All" || GROUP == "CPU"
cuda_testing() = LuxCUDA.functional() && (GROUP == "All" || GROUP == "CUDA")

if !@isdefined(MODES)
    const MODES = begin
        cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
        cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)

        modes = []
        cpu_testing() && push!(modes, cpu_mode)
        cuda_testing() && push!(modes, cuda_mode)

        modes
    end
end

# Some Helper Functions
function get_default_rng(mode::String)
    if mode == "CPU"
        return Random.default_rng()
    elseif mode == "CUDA"
        return CUDA.RNG()
    else
        error("Unknown mode: $mode")
    end
end

get_stable_rng(seed = 12345) = StableRNG(seed)

default_loss_function(model, ps, x, y) = mean(abs2, y .- model(x, ps))

train!(args...; kwargs...) = train!(default_loss_function, args...; kwargs...)

function train!(loss, model, ps, st, data; epochs = 10)
    m = Lux.Experimental.StatefulLuxLayer(model, ps, st)

    l1 = loss(m, ps, first(data)...)
    st_opt = Optimisers.setup(Adam(0.01f0), ps)
    for _ in 1:epochs, (x, y) in data
        _, gs, _, _ = Zygote.gradient(loss, m, ps, x, y)
        Optimisers.update!(st_opt, ps, gs)
    end
    l2 = loss(m, ps, first(data)...)

    return l2, l1
end
