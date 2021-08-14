using NeuralOperators
using DifferentialEquations
using Flux
using DiffEqFlux
using CUDA

if has_cuda()
    @info "CUDA is on"
    device = gpu
    CUDA.allowscalar(false)
else
    device = cpu
end


# model

tspan = (0.f0, 1.f0)

continuous_fop = NeuralODE(
    FourierOperator(64 => 64, 16) |> device,
    tspan,
    Tsit5(),
    save_everystep=false,
    reltol=1e-3,
    abstol=1e-3,
    save_start=false
) |> device

m = Chain(
    Dense(2, 64),
    continuous_fop,
    x -> reshape(device(x), size(x)[1:3]...),
    Dense(64, 1),
    Flux.flatten
) |> device

loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]


# data

n_train = 1800
n_test = 200
batchsize = 100
𝐱, 𝐲 = get_burgers_data(n=2048)


# training/testing set

𝐱_train, 𝐲_train = 𝐱[:, :, 1:n_train], 𝐲[:, 1:n_train]
loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=batchsize, shuffle=true)

𝐱_test, 𝐲_test = 𝐱[:, :, end-n_test+1:end], 𝐲[:, end-n_test+1:end]
loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=batchsize, shuffle=false)

function validate()
    validation_losses = [loss(device(𝐱), device(𝐲)) for (𝐱, 𝐲) in loader_test]
    @info "loss: $(sum(validation_losses)/length(loader_test))"
end


# train model

data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
call_back = Flux.throttle(validate, 5, leading=false, trailing=true)
Flux.@epochs 500 @time(Flux.train!(loss, Flux.params(m), data, opt, cb=call_back))
