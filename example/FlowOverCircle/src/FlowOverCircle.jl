module FlowOverCircle

using WaterLily, LinearAlgebra, ProgressMeter, MLUtils
using NeuralOperators, Flux
using CUDA

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

    model = MarkovNeuralOperator(ch=(1, 64, 64, 64, 64, 64, 1), modes=(24, 24), σ=gelu) |> device
    loader_train, loader_test = get_dataloader()
    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    optimiser = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
    loss_func(𝐱, 𝐲) = l₂loss(model(𝐱), 𝐲)

    function validate()
        validation_losses = [loss_func(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test]
        @info "loss: $(sum(validation_losses)/length(loader_test))"
    end

    @time begin
        for e in 1:50
            @warn "epoch $e"
            Flux.train!(loss_func, Flux.params(model), data, optimiser)
            validate()
        end
    end

    return learner
end

end # module
