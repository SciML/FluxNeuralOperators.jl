module DoublePendulum

using NeuralOperators
using Flux
using CUDA
using JLD2

include("data.jl")

__init__() = register_double_pendulum_chaotic()

function update_model!(model_file_path, model)
    model = cpu(model)
    jldsave(model_file_path; model)
    @warn "model updated!"
end

function train(; loss_bounds=[])
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = Chain(
        Dense(1, 64, gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, ), gelu),
        FourierOperator(64=>64, (12, )),
        Dense(64, 1)
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))

    loader_train, loader_test = get_dataloader()

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

    losses = Float32[]
    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "loss: $validation_loss"

        push!(losses, validation_loss)
        (losses[end] == minimum(losses)) && update_model!(joinpath(@__DIR__, "../model/model.jld2"), m)

        isempty(loss_bounds) && return
        if validation_loss < loss_bounds[1]
            @warn "change η"
            opt.os[2].eta /= 2
            popfirst!(loss_bounds)
        end
    end
    call_back = Flux.throttle(validate, 10, leading=false, trailing=true)

    Flux.@epochs 50 @time(Flux.train!(loss, params(m), data, opt, cb=call_back))
end

function get_model()
    f = jldopen(joinpath(@__DIR__, "../model/model.jld2"))
    model = f["model"]
    close(f)

    return model
end

end
