module NeuralOperators
    using Flux
    using FFTW
    using Tullio
    using CUDA
    using CUDAKernels
    using KernelAbstractions
    using Zygote
    using ChainRulesCore

    export DeepONet

    include("fourier.jl")
    include("model.jl")
    include("DeepONet.jl")
    include("subnets.jl")
end
