using Flux
using FFTW
using Tullio

export
    SpectralConv1d

struct SpectralConv1d{T,S}
    weight::T
    in_channel::S
    out_channel::S
    modes::S
end

function SpectralConv1d(
    ch::Pair{<:Integer,<:Integer},
    modes::Integer;
    init=Flux.glorot_uniform,
    T::DataType=Float32
)
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(out_chs, in_chs, modes)

    return SpectralConv1d(weights, in_chs, out_chs, modes)
end

Flux.@functor SpectralConv1d

function (m::SpectralConv1d)(𝐱::AbstractArray)
    𝐱_fft = rfft(𝐱, 1) # [x, in_chs, batch]
    𝐱_selected = 𝐱_fft[1:m.modes, :, :] # [modes, in_chs, batch]

    # [modes, out_chs, batch] <- [modes, in_chs, batch] [out_chs, in_chs, modes]
    @tullio 𝐱_weighted[m, o, b] := 𝐱_selected[m, i, b] * m.weight[o, i, m]

    d = size(𝐱, 1) ÷ 2 + 1 - m.modes
    𝐱_padded = cat(𝐱_weighted, zeros(Float32, d, size(𝐱)[2:end]...), dims=1)

    𝐱_out = irfft(𝐱_padded , size(𝐱, 1), 1)

    return 𝐱_out
end

# function FNO(modes::Integer, width::Integer)
#     return Chain(
#         PermutedDimsArray(Dense(2, width),(2,1,3)),
#         relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
#         relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
#         relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)),
#         PermutedDimsArray(relu(SpectralConv1d(width, width, modes) + Conv(width, width, 1)), (0, 2, 1)),
#         Dense(width, 128, relu),
#         Dense(128, 1)
#     )
# end

# loss(m::SpectralConv1d, x, x̂) = sum(abs2, x̂ .- m(x)) / len
