@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, LuxCUDA, AMDGPU, Zygote, Optimisers, Random, StableRNGs
using LuxTestUtils: @jet, @test_gradients

CUDA.allowscalar(false)

const BACKEND_GROUP = get(ENV, "BACKEND_GROUP", "All")

cpu_testing() = BACKEND_GROUP == "All" || BACKEND_GROUP == "CPU"
function cuda_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "CUDA") &&
           LuxDeviceUtils.functional(LuxCUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "All" || BACKEND_GROUP == "AMDGPU") &&
           LuxDeviceUtils.functional(LuxAMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("CPU", Array, LuxCPUDevice(), false))
    cuda_testing() && push!(modes, ("CUDA", CuArray, LuxCUDADevice(), true))
    amdgpu_testing() && push!(modes, ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true))
    modes
end

# Some Helper Functions
function get_default_rng(mode::String)
    dev = mode == "CPU" ? LuxCPUDevice() :
          mode == "CUDA" ? LuxCUDADevice() : mode == "AMDGPU" ? LuxAMDGPUDevice() : nothing
    rng = default_device_rng(dev)
    return rng isa TaskLocalRNG ? copy(rng) : deepcopy(rng)
end

train!(args...; kwargs...) = train!(MSELoss(), AutoZygote(), args...; kwargs...)

function train!(loss, backend, model, ps, st, data; epochs=10)
    l1 = loss(model, ps, st, first(data))

    tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, _, _, tstate = Lux.Experimental.single_train_step!(backend, loss, (x, y), tstate)
    end

    l2 = loss(model, ps, st, first(data))

    return l2, l1
end

export @jet, @test_gradients, check_approx
export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, get_default_rng,
       train!

end
