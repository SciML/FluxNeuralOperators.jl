using Zygote
using Flux
using CUDA
using FFTW
using Tullio

if has_cuda()
    @info "CUDA is on"
    device = gpu
    CUDA.allowscalar(true)
else
    device = cpu
end

function t(𝐱)
    @tullio 𝐱ᵀ[a, b, c] := 𝐱[b, a, c]

    return 𝐱ᵀ
end

m = Chain(
    Dense(2, 5),
    t,
    x->Zygote.hook(real, x),
    x->real(fft(x, 1)),
    t,
    Dense(5, 5),
    t,
    x->Zygote.hook(real, x),
    x->real(ifft(x, 1)),
    t,
    x->sum(x)
) |> device

loss(x, y) = Flux.mse(m(x), y)

data = [(rand(Float32, 2, 100, 10), rand(Float32, 10))] |> device
Flux.train!(loss, params(m), data, Flux.ADAM())
