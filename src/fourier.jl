export
    SpectralConv1d,
    FourierOperator,
    FNO

struct SpectralConv1d{T, S}
    weight::T
    in_channel::S
    out_channel::S
    modes::S
    σ
end

function c_glorot_uniform(dims...)
    return Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...) * im
end

function SpectralConv1d(
    ch::Pair{<:Integer, <:Integer},
    modes::Integer,
    σ=identity;
    init=c_glorot_uniform,
    T::DataType=ComplexF32
)
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(out_chs, in_chs, modes)

    return Chain(
        x -> Zygote.hook(real, x),
        SpectralConv1d(weights, in_chs, out_chs, modes, σ),
    )
end

Flux.@functor SpectralConv1d

t(𝐱) = @tullio 𝐱ᵀ[i, j, k] := 𝐱[j, i, k]
ein_mul(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[o, i, m]

function (m::SpectralConv1d)(𝐱::AbstractArray)
    𝐱ᵀ = t(𝐱) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_fft = fft(𝐱ᵀ, 1) # [x, in_chs, batch]
    𝐱_selected = 𝐱_fft[1:m.modes, :, :] # [modes, in_chs, batch]

    # [modes, out_chs, batch] <- [modes, in_chs, batch] * [out_chs, in_chs, modes]
    𝐱_weighted = ein_mul(𝐱_selected, m.weight)

    s = size(𝐱_weighted)[2:end]
    d = size(𝐱ᵀ, 1) - m.modes
    𝐱_padded = cat(𝐱_weighted, zeros(ComplexF32, d, s...), dims=1)

    𝐱_out = ifft(𝐱_padded, 1) # [x, out_chs, batch]
    𝐱_outᵀ = t(𝐱_out) # [out_chs, x, batch] <- [x, out_chs, batch]

    return m.σ.(real(𝐱_outᵀ))
end

function FourierOperator(
    ch::Pair{<:Integer, <:Integer},
    modes::Integer,
    σ=identity
)
    return Chain(
        Parallel(+,
            Dense(ch.first, ch.second),
            SpectralConv1d(ch, modes)
        ),
        x -> σ.(x)
    )
end
