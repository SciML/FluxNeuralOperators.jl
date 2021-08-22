module Burgers

using NeuralOperators
using Flux
using CUDA

include("data.jl")

__init__() = register_burgers()

function train()
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    modes = (16, )
    ch = 64 => 64
    σ = gelu
    m = Chain(
        Dense(2, 64),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes),
        Dense(64, 128, σ),
        Dense(128, 1),
        flatten
    ) |> device
    loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

    n_train = 1800
    n_test = 200
    batchsize = 100
    𝐱, 𝐲 = get_burgers_data(n=2048)

    𝐱_train, 𝐲_train = 𝐱[:, :, 1:n_train], 𝐲[:, 1:n_train]
    loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=batchsize, shuffle=true)

    𝐱_test, 𝐲_test = 𝐱[:, :, end-n_test+1:end], 𝐲[:, end-n_test+1:end]
    loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=batchsize, shuffle=false)

    function validate()
        validation_losses = [loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test]
        @info "loss: $(sum(validation_losses)/length(loader_test))"
    end

    data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
    opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
    call_back = Flux.throttle(validate, 5, leading=false, trailing=true)
    Flux.@epochs 500 @time(Flux.train!(loss, params(m), data, opt, cb=call_back))
end

end
