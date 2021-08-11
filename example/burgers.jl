using NeuralOperators
using Flux
using CUDA

if has_cuda()
    @info "CUDA is on"
    device = gpu
    CUDA.allowscalar(false)
else
    device = cpu
end

m = FourierNeuralOperator() |> device
loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- m(𝐱)) / size(𝐱)[end]

𝐱, 𝐲 = get_burgers_data(n=2048)

n_train = 2000
𝐱_train, 𝐲_train = 𝐱[:, :, 1:n_train], 𝐲[:, 1:n_train]
loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=20, shuffle=true)

n_test = 40
𝐱_test, 𝐲_test = 𝐱[:, :, end-n_test+1:end], 𝐲[:, end-n_test+1:end]
loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=20, shuffle=false)

function loss_test()
    l = 0
    for (𝐱, 𝐲) in loader_test
        l += loss(𝐱, 𝐲)
    end
    @info "loss: $(l/length(loader_test))"
end

data = [(𝐱, 𝐲) for (𝐱, 𝐲) in loader_train] |> device
opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
Flux.@epochs 500 @time(Flux.train!(loss, params(m), data, opt))
