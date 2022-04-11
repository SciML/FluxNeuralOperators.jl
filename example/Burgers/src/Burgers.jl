module Burgers

using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using CUDA, FluxTraining, BSON

include("Burgers_deeponet.jl")

function register_burgers()
    register(DataDep(
        "Burgers",
        """
        Burgers' equation dataset from
        [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
        """,
        "http://www.med.cgu.edu.tw/NeuralOperators/Burgers_R10.zip",
        "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd",
        post_fetch_method=unpack
    ))
end

function get_data(; n=2048, Δsamples=2^3, grid_size=div(2^13, Δsamples), T=Float32)
    file = matopen(joinpath(datadep"Burgers", "burgers_data_R10.mat"))
    x_data = T.(collect(read(file, "a")[1:n, 1:Δsamples:end]'))
    y_data = T.(collect(read(file, "u")[1:n, 1:Δsamples:end]'))
    close(file)

    x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
    x_loc_data[1, :, :] .= reshape(repeat(LinRange(0, 1, grid_size), n), (grid_size, n))
    x_loc_data[2, :, :] .= x_data

    return x_loc_data, reshape(y_data, 1, :, n)
end

function get_dataloader(; ratio::Float64=0.9, batchsize=100)
    𝐱, 𝐲 = get_data(n=2048)
    data_train, data_test = splitobs((𝐱, 𝐲), at=ratio)

    loader_train = DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_test = DataLoader(data_test, batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end

__init__() = register_burgers()

function train(; cuda=true, η₀=1f-3, λ=1f-4, epochs=500)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = FourierNeuralOperator(ch=(2, 64, 64, 64, 64, 64, 128, 1), modes=(16, ), σ=gelu)
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.ADAM(η₀))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
    )

    fit!(learner, epochs)

    return learner
end

end
