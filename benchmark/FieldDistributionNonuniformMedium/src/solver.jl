using NeuralOperators
using Flux
using CUDA

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @warn "model updated!"
end

function train(η, batchsize=10)
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = Chain(
        Dense(3, 64),
        FourierOperator(64=>64, (16, 16), gelu),
        FourierOperator(64=>64, (16, 16), gelu),
        FourierOperator(64=>64, (16, 16), gelu),
        FourierOperator(64=>64, (16, 16)),
        Dense(64, 128, gelu),
        Dense(128, 1),
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(η))

    loader_train, loader_test = get_dataloader(batchsize=batchsize)

    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(@__DIR__, "../model/model.jld2"), m)
    end
    # call_back = Flux.throttle(validate, 10, leading=false, trailing=true)

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    for e in 1:200
        @info "Epoch $e\n η: $(opt.os[2].eta)"
        @time begin
            Flux.train!(loss, params(m), data, opt)
            validate()
        end
        (e%10 == 0) && (opt.os[2].eta /= 2)
    end

    return m
end

function get_model()
    f = jldopen(joinpath(@__DIR__, "../model/model.jld2"))
    model = f["model"]
    close(f)

    return model
end
