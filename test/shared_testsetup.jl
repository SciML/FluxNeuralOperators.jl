@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, Zygote, Optimisers, Random, StableRNGs
using LuxTestUtils: @jet, @test_gradients

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "All"))

if BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda"
    using LuxCUDA
end

if BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu"
    using AMDGPU
end

cpu_testing() = BACKEND_GROUP == "all" || BACKEND_GROUP == "cpu"
function cuda_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") &&
           LuxDeviceUtils.functional(LuxCUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           LuxDeviceUtils.functional(LuxAMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("CPU", Array, LuxCPUDevice(), false))
    cuda_testing() && push!(modes, ("CUDA", CuArray, LuxCUDADevice(), true))
    amdgpu_testing() && push!(modes, ("AMDGPU", ROCArray, LuxAMDGPUDevice(), true))
    modes
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
export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, train!

end
