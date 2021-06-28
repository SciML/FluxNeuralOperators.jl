using FFTW
using Flux
using CUDA
CUDA.allowscalar(false)

##################
# SpectralConv1d #
##################

mutable struct SpectralConv1d
    𝐰::Matrix{Float32}
    in_channels::Int32
    out_channels::Int32
    modes::Int32
end

function SpectralConv1d(in_channels::Integer, out_channels::Integer, modes::Integer)
    scale = 1 / (in_channels * out_channels)

    return SpectralConv1d(
        scale * rand(in_channels, out_channels, modes),
        in_channels,
        out_channels,
        modes
    )
end

Flux.@functor SpectralConv1d

function (m::SpectralConv1d)(x)
    fft!(x)
    out_ft = zeros(size(x, 1), m.out_channels, div(size(x)[end], 2)+1)
    out_ft[:, :, 1:m.modes] = x[:, :, 1:m.modes] * m.𝐰

    return ifft(out_ft[1:size(x)[end]])
end

loss(m::SpectralConv1d, x, x̂) = sum(abs2, x̂ .- m(x)) / len

#######
# FNO #
#######

function FNO(modes::Integer, width::Integer)
    return Chain(
        PermutedDimsArray(Dense(2, width),(2,1,3)),
        relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
        relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
        relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
        PermutedDimsArray(relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)), (0, 2, 1)),
        Dense(width, 128, relu),
        Dense(128, 1)
    )
end
