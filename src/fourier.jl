export
    SpectralConv1d,
    FourierOperator

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im

struct SpectralConv1d{T, S}
    weight::T
    in_channel::S
    out_channel::S
    modes::S
    σ
end

function SpectralConv1d(
    ch::Pair{<:Integer,<:Integer},
    modes::Integer,
    σ=identity;
    init=c_glorot_uniform,
    T::DataType=ComplexF32
)
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(out_chs, in_chs, modes)

    return SpectralConv1d(weights, in_chs, out_chs, modes, σ)
end

Flux.@functor SpectralConv1d

function (m::SpectralConv1d)(𝐱::AbstractArray)
    𝐱_fft = fft(𝐱, 2) # [in_chs, x, batch]
    𝐱_selected = 𝐱_fft[:, 1:m.modes, :] # [in_chs, modes, batch]

    # [out_chs, modes, batch] <- [in_chs, modes, batch] [out_chs, in_chs, modes]
    @tullio 𝐱_weighted[o, m, b] := 𝐱_selected[i, m, b] * m.weight[o, i, m]

    s = size(𝐱_weighted)
    d = size(𝐱, 2) - m.modes
    𝐱_padded = cat(𝐱_weighted, zeros(ComplexF32, s[1], d, s[3:end]...), dims=2)

    𝐱_out = ifft(𝐱_padded, 2)

    return m.σ.(𝐱_out)
end

function FourierOperator(
    ch::Pair{<:Integer,<:Integer},
    modes::Integer,
    σ=identity
)
    return Chain(
        Parallel(+,
            Dense(ch.first, ch.second, init=c_glorot_uniform),
            SpectralConv1d(ch, modes)
        ),
        x -> σ.(x)
    )
end
