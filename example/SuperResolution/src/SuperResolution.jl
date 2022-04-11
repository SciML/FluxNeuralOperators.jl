module SuperResolution

using WaterLily, LinearAlgebra, ProgressMeter, MLUtils
using NeuralOperators, Flux
using CUDA, FluxTraining, BSON

function circle(n, m; Re=250) # copy from [WaterLily](https://github.com/weymouth/WaterLily.jl)
    # Set physical parameters
    U, R, center = 1., m/8., [m/2, m/2]
    ν = U * R / Re

    body = AutoBody((x,t) -> LinearAlgebra.norm2(x .- center) - R)
    Simulation((n+2, m+2), [U, 0.], R; ν, body)
end

function gen_data(ts::AbstractRange; resolution=2)
    @info "gen data with $(resolution)x resolution... "
    p = Progress(length(ts))

    n, m = resolution * 3(2^5), resolution * 2^6
    circ = circle(n, m)

    𝐩s = Array{Float32}(undef, 1, n, m, length(ts))
    for (i, t) in enumerate(ts)
        sim_step!(circ, t)
        𝐩s[1, :, :, i] .= Float32.(circ.flow.p)[2:end-1, 2:end-1]

        next!(p)
    end

    return 𝐩s
end

function get_dataloader(; ts::AbstractRange=LinRange(100, 11000, 10000), ratio::Float64=0.95, batchsize=100)
    data = gen_data(ts, resolution=1)
    data_train, data_validate = splitobs(shuffleobs((𝐱=data[:, :, :, 1:end-1], 𝐲=data[:, :, :, 2:end])), at=ratio)

    data = gen_data(ts, resolution=2)
    _, data_super_res = splitobs(shuffleobs((𝐱=data[:, :, :, 1:end-1], 𝐲=data[:, :, :, 2:end])), at=ratio)

    loader_train = DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_validate = DataLoader(data_validate, batchsize=batchsize, shuffle=false)
    loader_super_res = DataLoader(data_super_res, batchsize=batchsize, shuffle=false)

    return (training=loader_train, validation=loader_validate, super_res=loader_super_res)
end

struct SuperResPhase<:FluxTraining.AbstractValidationPhase end

FluxTraining.phasedataiter(::SuperResPhase) = :super_res

function FluxTraining.step!(learner, phase::SuperResPhase, batch)
    xs, ys = batch
    FluxTraining.runstep(learner, phase, (xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end

function fit!(learner, nepochs::Int, (loader_train, loader_validate, loader_super_res))
    for i in 1:nepochs
        epoch!(learner, TrainingPhase(), loader_train)
        epoch!(learner, ValidationPhase(), loader_validate)
        epoch!(learner, SuperResPhase(), loader_super_res)
    end
end

function fit!(learner, nepochs::Int)
    fit!(learner, nepochs, (learner.data.training, learner.data.validation, learner.data.super_res))
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

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

end # module
