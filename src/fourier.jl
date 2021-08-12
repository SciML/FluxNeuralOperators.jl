export
    SpectralConv1d,
    FourierOperator

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

    return SpectralConv1d(weights, in_chs, out_chs, modes, σ)
end

Flux.@functor SpectralConv1d

ein_mul(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[o, i, m]

gen_pad(args...; T=ComplexF32) = zeros(T, args...)

function (m::SpectralConv1d)(𝐱::AbstractArray)
    𝐱ᵀ = permutedims(Zygote.hook(real, 𝐱), [2, 1, 3]) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_fft = fft(𝐱ᵀ, 1) # [x, in_chs, batch]

    # [modes, out_chs, batch] <- [modes, in_chs, batch] * [out_chs, in_chs, modes]
    𝐱_weighted = ein_mul(𝐱_fft[1:m.modes, :, :], m.weight)
    pad = gen_pad(size(𝐱_fft, 1)-m.modes, size(𝐱_weighted)[2:end]...)
    𝐱_padded = cat(𝐱_weighted, pad, dims=1) # [x, out_chs, batch] <- [modes, out_chs, batch]

    𝐱_out = ifft(𝐱_padded, 1) # [x, out_chs, batch]
    𝐱_outᵀ = permutedims(real(𝐱_out), [2, 1, 3]) # [out_chs, x, batch] <- [x, out_chs, batch]

    return m.σ.(𝐱_outᵀ)
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
