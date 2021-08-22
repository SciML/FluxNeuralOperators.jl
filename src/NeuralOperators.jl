module NeuralOperators
    using Flux
    using FFTW
    using Tullio
    using CUDA
    using CUDAKernels
    using KernelAbstractions
    using Zygote
    using ChainRulesCore

    include("fourier.jl")
    include("model.jl")
end
