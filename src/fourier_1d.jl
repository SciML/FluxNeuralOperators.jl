using MAT
using FFTW
using Flux
using CUDA
CUDA.allowscalar(false)

export get_data, preprocess

##############
# preprocess #
##############

function get_data()
    n_train = 1000
    subsampling_rate = 2^3
    total_grid_size = div(2^13, subsampling_rate)

    file = matopen("data/burgers_data_R10.mat")
    x_data = read(file, "a")[1:n_train, 1:subsampling_rate:end]
    y_data = read(file, "u")[1:n_train, 1:subsampling_rate:end]
    close(file)

    grid = reshape(repeat(LinRange(0, 1, total_grid_size), n_train),(total_grid_size, n_train))'
    x_data = cat(x_data, grid, dims=3)

    return x_data, y_data
end

function preprocess()
end

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
    x = fft(x)
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
