@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, LuxCUDA, LuxAMDGPU, Zygote, Optimisers, Random, StableRNGs, Statistics
using LuxTestUtils: @jet, @test_gradients

CUDA.allowscalar(false)

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

cpu_testing() = BACKEND_GROUP == "All" || BACKEND_GROUP == "CPU"
cuda_testing() = (BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA") && LuxCUDA.functional()
function amdgpu_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU") && LuxAMDGPU.functional()
end

const MODES = begin
    # Mode, Array Type, Device Function, GPU?
    cpu_mode = ("CPU", Array, LuxCPUDevice(), false)
    cuda_mode = ("CUDA", CuArray, LuxCUDADevice(), true)
    amdgpu_mode = ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true)

    modes = []
    cpu_testing() && push!(modes, cpu_mode)
    cuda_testing() && push!(modes, cuda_mode)
    amdgpu_testing() && push!(modes, amdgpu_mode)

    modes
end

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "CPU" ? LuxCPUDevice() :
          mode == "CUDA" ? LuxCUDADevice() : mode == "AMDGPU" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

get_stable_rng(seed=12345) = StableRNG(seed)

default_loss_function(model, ps, x, y) = mean(abs2, y .- model(x, ps))

train!(args...; kwargs...) = train!(default_loss_function, args...; kwargs...)

function train!(loss, model, ps, st, data; epochs=10)
    m = StatefulLuxLayer(model, ps, st)

    l1 = loss(m, ps, first(data)...)
    st_opt = Optimisers.setup(Adam(0.01f0), ps)
    for _ in 1:epochs, (x, y) in data
        _, gs, _, _ = Zygote.gradient(loss, m, ps, x, y)
        Optimisers.update!(st_opt, ps, gs)
    end
    l2 = loss(m, ps, first(data)...)

    return l2, l1
end

export @jet, @test_gradients, check_approx
export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
       get_stable_rng, train!

end
