module NeuralOperators
    using DataDeps
    using Fetch
    using MAT

    using Flux
    using FFTW
    using Tullio
    using CUDA
    using CUDAKernels
    using KernelAbstractions
    using Zygote

    function __init__()
        register_datasets()
    end

    include("data.jl")
    include("fourier.jl")
    include("model.jl")
end
