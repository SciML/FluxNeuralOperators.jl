module FlowOverCircle

using WaterLily, LinearAlgebra, ProgressMeter, MLUtils
using NeuralOperators, Flux, GeometricFlux, Graphs
using CUDA, FluxTraining, BSON
using GeometricFlux.GraphSignals: generate_grid

function circle(n, m; Re = 250) # copy from [WaterLily](https://github.com/weymouth/WaterLily.jl)
    # Set physical parameters
    U, R, center = 1.0, m / 8.0, [m / 2, m / 2]
    ν = U * R / Re

    body = AutoBody((x, t) -> LinearAlgebra.norm2(x .- center) - R)
    Simulation((n + 2, m + 2), [U, 0.0], R; ν, body)
end

function gen_data(ts::AbstractRange)
    @info "gen data... "
    p = Progress(length(ts))

    n, m = 3(2^5), 2^6
    circ = circle(n, m)

    𝐩s = Array{Float32}(undef, 1, n, m, length(ts))
    for (i, t) in enumerate(ts)
        sim_step!(circ, t)
        𝐩s[1, :, :, i] .= Float32.(circ.flow.p)[2:(end - 1), 2:(end - 1)]

        next!(p)
    end

    return 𝐩s
end

function get_mno_dataloader(; ts::AbstractRange = LinRange(100, 11000, 10000),
                            ratio::Float64 = 0.95, batchsize = 100)
    data = gen_data(ts)
    𝐱, 𝐲 = data[:, :, :, 1:(end - 1)], data[:, :, :, 2:end]
    n = length(ts) - 1

    data_train, data_test = splitobs(shuffleobs((𝐱, 𝐲)), at = ratio)

    loader_train = DataLoader(data_train, batchsize = batchsize, shuffle = true)
    loader_test = DataLoader(data_test, batchsize = batchsize, shuffle = false)

    return loader_train, loader_test
end

function train_mno(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 50)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = MarkovNeuralOperator(ch = (1, 64, 64, 64, 64, 64, 1), modes = (24, 24),
                                 σ = gelu)
    data = get_mno_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss

    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device),
                      Checkpointer(joinpath(@__DIR__, "../model/")))

    fit!(learner, epochs)

    return learner
end

function get_gno_dataloader(; ts::AbstractRange = LinRange(100, 11000, 10000),
                            ratio::Float64 = 0.95, batchsize = 8)
    data = gen_data(ts)
    𝐱, 𝐲 = data[:, :, :, 1:(end - 1)], data[:, :, :, 2:end]
    n = length(ts) - 1

    # generate graph
    graph = Graphs.grid(size(data)[2:3])

    # add grid coordinates
    grid = generate_coordinates(𝐱[1, :, :, 1])
    grid = repeat(grid, outer = (1, 1, 1, n))
    𝐱 = vcat(𝐱, grid)

    # flatten
    𝐱, 𝐲 = reshape(𝐱, size(𝐱, 1), :, n), reshape(𝐲, 1, :, n)

    fg = FeaturedGraph(graph, nf = 𝐱, pf = 𝐱)
    data_train, data_test = splitobs(shuffleobs((fg, 𝐲)), at = ratio)

    loader_train = DataLoader(data_train, batchsize = batchsize, shuffle = true)
    loader_test = DataLoader(data_test, batchsize = batchsize, shuffle = false)

    return loader_train, loader_test
end

function train_gno(; cuda = true, η₀ = 1.0f-3, λ = 1.0f-4, epochs = 50)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    grid_dim = 2
    edge_dim = 2(grid_dim + 1)
    model = Chain(GraphParallel(node_layer = Dense(grid_dim + 1, 16)),
                  GraphKernel(Dense(edge_dim, abs2(16), gelu), 16),
                  GraphKernel(Dense(edge_dim, abs2(16), gelu), 16),
                  GraphKernel(Dense(edge_dim, abs2(16), gelu), 16),
                  GraphKernel(Dense(edge_dim, abs2(16), gelu), 16),
                  node_feature,
                  Dense(16, 1))

    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.Adam(η₀))
    loss_func = l₂loss
    data = get_gno_dataloader()
    learner = Learner(model, data, optimiser, loss_func,
                      ToDevice(device, device),
                      Checkpointer(joinpath(@__DIR__, "../model/")))

    fit!(learner, epochs)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

end # module
