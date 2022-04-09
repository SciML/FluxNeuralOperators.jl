module FlowOverCircle

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

function get_dataloader(; ts::AbstractRange=LinRange(100, 11000, 10000), ratio::Float64=0.95, batchsize=100)
    data = gen_data(ts)
    data_train, data_test = splitobs((𝐱=data[:, :, :, 1:end-1], 𝐲=data[:, :, :, 2:end]), at=ratio)

    loader_train = Flux.DataLoader(data_train, batchsize=batchsize, shuffle=true)
    loader_test = Flux.DataLoader(data_test, batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end

function train()
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    model = Chain(
        Dense(1, 64),
        OperatorKernel(64=>64, (24, 24), FourierTransform, gelu),
        OperatorKernel(64=>64, (24, 24), FourierTransform, gelu),
        OperatorKernel(64=>64, (24, 24), FourierTransform, gelu),
        OperatorKernel(64=>64, (24, 24), FourierTransform, gelu),
        Dense(64, 1),
    )
    data = get_dataloader()
    optimiser = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
    loss_func = l₂loss

    learner = Learner(
        model, data, optimiser, loss_func,
        ToDevice(device, device),
        Checkpointer(joinpath(@__DIR__, "../model/"))
    )

    fit!(learner, 50)

    return learner
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

end # module
