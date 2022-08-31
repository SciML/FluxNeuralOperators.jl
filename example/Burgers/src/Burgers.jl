module Burgers

using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using CUDA, FluxTraining, BSON
import Flux: params
using BSON: @save, @load

include("Burgers_deeponet.jl")

function register_burgers()
    register(DataDep("Burgers",
                     """
                     Burgers' equation dataset from
                     [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)

                     mapping between initial conditions to the solutions at the last point of time evolition in some function space. 
                     u(x,0) -> u(x, time_end):

                     * `a`: initial conditions u(x,0)
                     * `u`: solutions u(x,t_end)
                     """,

                     "http://www.med.cgu.edu.tw/NeuralOperators/Burgers_R10.zip",
                     "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd",
                     post_fetch_method = unpack))
end

function get_data(; n = 2048, Δsamples = 2^3, grid_size = div(2^13, Δsamples), T = Float32)
    file = matopen(joinpath(datadep"Burgers", "burgers_data_R10.mat"))
    x_data = T.(collect(read(file, "a")[1:n, 1:Δsamples:end]'))
    y_data = T.(collect(read(file, "u")[1:n, 1:Δsamples:end]'))
    close(file)

    x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
    x_loc_data[1, :, :] .= reshape(repeat(LinRange(0, 1, grid_size), n), (grid_size, n))
    x_loc_data[2, :, :] .= x_data

    return x_loc_data, reshape(y_data, 1, :, n)
end

function get_dataloader(; ratio::Float64 = 0.9, batchsize = 100)
    𝐱, 𝐲 = get_data(n = 2048)
    data_train, data_test = splitobs((𝐱, 𝐲), at = ratio)

    loader_train = DataLoader(data_train, batchsize = batchsize, shuffle = true)
    loader_test = DataLoader(data_test, batchsize = batchsize, shuffle = false)

    return loader_train, loader_test
end

__init__() = register_burgers()

function train(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 500)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = FourierNeuralOperator(ch = (2, 64, 64, 64, 64, 64, 128, 1), modes = (16,),
                                  σ = gelu)
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device))

    fit!(learner, epochs)
    model = learner.model |> cpu
    @save "model/model_burger.bson" model

    return learner
end

function train_nomad(; n = 300, cuda = true, learning_rate = 0.001, epochs = 400)
    if cuda && has_cuda()
        @info "Training on GPU"
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    x, y = get_data_don(n = n)

    xtrain = x[1:280, :]'
    ytrain = y[1:280, :]

    xval = x[(end - 19):end, :]' |> device
    yval = y[(end - 19):end, :] |> device

    # grid = collect(range(0, 1, length=1024)') |> device
    grid = rand(collect(0:0.001:1), (280, 1024)) |> device
    gridval = rand(collect(0:0.001:1), (20, 1024)) |> device

    opt = Adam(learning_rate)

    m = NOMAD((1024, 1024), (2048, 1024), gelu, gelu) |> device

    loss(X, y, sensor) = Flux.Losses.mse(m(X, sensor), y)
    evalcb() = @show(loss(xval, yval, gridval))

    data = [(xtrain, ytrain, grid)] |> device
    Flux.@epochs epochs Flux.train!(loss, params(m), data, opt, cb = evalcb)
    ỹ = m(xval |> device, gridval |> device)

    diffvec = vec(abs.(cpu(yval) .- cpu(ỹ)))
    mean_diff = sum(diffvec) / length(diffvec)
    return mean_diff
end


function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

end
