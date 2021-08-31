module DoublePendulum

using NeuralOperators
using Flux
using CUDA

include("data.jl")

__init__() = register_double_pendulum_chaotic()

function train()
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    m = Chain(
        FourierOperator(6=>6, (16, ), gelu),
        FourierOperator(6=>6, (16, ), gelu),
        FourierOperator(6=>6, (16, ), gelu),
        FourierOperator(6=>6, (16, )),
    ) |> device

    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    loader_train, loader_test = get_dataloader()

    function validate()
        validation_losses = [loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test]
        @info "loss: $(sum(validation_losses)/length(loader_test))"
    end

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-4))
    call_back = Flux.throttle(validate, 0.5, leading=false, trailing=true)
    Flux.@epochs 500 @time(Flux.train!(loss, params(m), data, opt, cb=call_back))
end

end
