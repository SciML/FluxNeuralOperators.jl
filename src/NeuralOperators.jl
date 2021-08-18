module NeuralOperators
    using DataDeps
    using Fetch
    using MAT
    using StatsBase

    using Flux
    using FFTW
    using Tullio
    using CUDA
    using CUDAKernels
    using KernelAbstractions
    using Zygote
    using ChainRulesCore

    function __init__()
        register_datasets()
    end

    include("data.jl")
    include("fourier.jl")
    include("model.jl")
end
