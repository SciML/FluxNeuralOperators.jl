@testsetup module SharedTestSetup
import Reexport: @reexport

@reexport using Lux, Zygote, Optimisers, Random, StableRNGs, LuxTestUtils
using MLDataDevices

LuxTestUtils.jet_target_modules!(["NeuralOperators", "Lux", "LuxLib"])

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
           MLDataDevices.functional(CUDADevice)
end
function amdgpu_testing()
    return (BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") &&
           MLDataDevices.functional(AMDGPUDevice)
end

const MODES = begin
    modes = []
    cpu_testing() && push!(modes, ("CPU", Array, CPUDevice(), false))
    cuda_testing() && push!(modes, ("CUDA", CuArray, CUDADevice(), true))
    amdgpu_testing() && push!(modes, ("AMDGPU", ROCArray, AMDGPUDevice(), true))
    modes
end

train!(args...; kwargs...) = train!(MSELoss(), AutoZygote(), args...; kwargs...)

function train!(loss, backend, model, ps, st, data; epochs=10)
    l1 = loss(model, ps, st, first(data))

    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, _, _, tstate = Training.single_train_step!(backend, loss, (x, y), tstate)
    end

    l2 = loss(model, ps, st, first(data))

    return l2, l1
end

export check_approx
export BACKEND_GROUP, MODES, cpu_testing, cuda_testing, amdgpu_testing, train!

end
