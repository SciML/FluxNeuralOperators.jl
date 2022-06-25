module NeuralOperators
    using Flux
    using FFTW
    using Tullio
    using CUDA
    using CUDAKernels
    using KernelAbstractions
    using Zygote
    using ChainRulesCore
    using GeometricFlux
    using Statistics

    export DeepONet, NOMAD

    include("Transform/Transform.jl")
    include("operator_kernel.jl")
    include("loss.jl")
    include("model.jl")
    include("DeepONet.jl")
    include("NOMAD.jl")
    include("subnets.jl")
end
