module FlowOverCircle

using WaterLily, LinearAlgebra, ProgressMeter, MLUtils
using NeuralOperators, Flux, GeometricFlux, Graphs
using CUDA, FluxTraining, BSON

function circle(n, m; Re=250) # copy from [WaterLily](https://github.com/weymouth/WaterLily.jl)
    # Set physical parameters
    U, R, center = 1., m/8., [m/2, m/2]
    ν = U * R / Re

    body = AutoBody((x,t) -> LinearAlgebra.norm2(x .- center) - R)
    Simulation((n+2, m+2), [U, 0.], R; ν, body)
end

function gen_data(ts::AbstractRange)
    @info "gen data... "
    p = Progress(length(ts))

    n, m = 3(2^5), 2^6
    circ = circle(n, m)

    𝐩s = Array{Float32}(undef, 1, n, m, length(ts))
    for (i, t) in enumerate(ts)
        sim_step!(circ, t)
        𝐩s[1, :, :, i] .= Float32.(circ.flow.p)[2:end-1, 2:end-1]

        next!(p)
    end

    return 𝐩s
end

function get_dataloader(; ts::AbstractRange=LinRange(100, 11000, 10000), ratio::Float64=0.95, batchsize=100, flatten=false)
    data = gen_data(ts)
    𝐱, 𝐲 = data[:, :, :, 1:end-1], data[:, :, :, 2:end]
    n = length(ts) - 1

    if flatten
        𝐱, 𝐲 = reshape(𝐱, 1, :, n), reshape(𝐲, 1, :, n)
    end

    data_train, data_test = splitobs(shuffleobs((𝐱, 𝐲)), at=ratio)

    loader_train = DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_test = DataLoader(data_test, batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end

function train(; cuda=true, η₀=1f-3, λ=1f-4, epochs=50)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    model = MarkovNeuralOperator(ch=(1, 64, 64, 64, 64, 64, 1), modes=(24, 24), σ=gelu)
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.ADAM(η₀))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
        Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, epochs)

    return learner
end

function train_gno(; cuda=true, η₀=1f-3, λ=1f-4, epochs=50)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    featured_graph = FeaturedGraph(grid([96, 64]))
    model = Chain(
        Dense(1, 16),
        WithGraph(featured_graph, GraphKernel(Dense(2*16, 16, gelu), 16)),
        WithGraph(featured_graph, GraphKernel(Dense(2*16, 16, gelu), 16)),
        WithGraph(featured_graph, GraphKernel(Dense(2*16, 16, gelu), 16)),
        WithGraph(featured_graph, GraphKernel(Dense(2*16, 16, gelu), 16)),
        Dense(16, 1),
    )
    data = get_dataloader(batchsize=16, flatten=true)
    optimiser = Flux.Optimiser(WeightDecay(λ), Flux.ADAM(η₀))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
        Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, epochs)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

end # module
