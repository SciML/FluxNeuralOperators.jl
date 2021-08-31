module DoublePendulum

using NeuralOperators
using Flux
using CUDA

include("data.jl")

__init__() = register_double_pendulum_chaotic()

function train(; loss_bounds=[1, 0.2, 0.1, 0.05, 0.02])
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = Chain(
        FourierOperator(6=>64, (16, ), relu),
        FourierOperator(64=>64, (16, ), relu),
        FourierOperator(64=>64, (16, ), relu),
        FourierOperator(64=>6, (16, )),
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-2))

    loader_train, loader_test = get_dataloader()

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device

    function validate()
        validation_loss = sum(loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test)/length(loader_test)
        @info "loss: $validation_loss"

        isempty(loss_bounds) && return
        if validation_loss < loss_bounds[1]
            @warn "change η"
            opt.os[2].eta /= 2
            popfirst!(loss_bounds)
        end
    end

    call_back = Flux.throttle(validate, 1, leading=false, trailing=true)
    Flux.@epochs 300 @time(Flux.train!(loss, params(m), data, opt, cb=call_back))
end

end
