export
    SpectralConv,
    FourierOperator

struct SpectralConv{T, S, N}
    weight::T
    in_channel::S
    out_channel::S
    modes::NTuple{N, S}
    ndim::S
    σ
end

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...)*im

function SpectralConv(
    ch::Pair{S, S},
    modes::NTuple{N, S},
    σ=identity;
    init=c_glorot_uniform,
    T::DataType=ComplexF32
) where {S<:Integer, N}
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(out_chs, in_chs, modes...)

    return SpectralConv(weights, in_chs, out_chs, modes, N, σ)
end

Flux.@functor SpectralConv

spectral_conv(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[o, i, m] # TODO: extend `m` to n-dim

function (m::SpectralConv)(𝐱::AbstractArray)
    𝐱ᵀ = permutedims(Zygote.hook(real, 𝐱), (m.ndim+1, 1:m.ndim..., m.ndim+2)) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_fft = fft(𝐱ᵀ, 1:m.ndim) # [x, in_chs, batch]

    # [modes, out_chs, batch] <- [modes, in_chs, batch] * [out_chs, in_chs, modes]
    ranges = [1:dim_modes for dim_modes in m.modes]
    𝐱_weighted = spectral_conv(view(𝐱_fft, ranges..., :, :), m.weight)

    # [x, out_chs, batch] <- [modes, out_chs, batch]
    pad = zeros(ComplexF32, (collect(size(𝐱_fft)[1:m.ndim])-collect(m.modes))..., size(𝐱_weighted)[end-1:end]...)
    𝐱_padded = cat(𝐱_weighted, pad, dims=1:m.ndim)

    𝐱_out = ifft(𝐱_padded, 1:m.ndim) # [x, out_chs, batch]
    𝐱_outᵀ = permutedims(real(𝐱_out), (2:m.ndim+1..., 1, m.ndim+2)) # [out_chs, x, batch] <- [x, out_chs, batch]

    return m.σ.(𝐱_outᵀ)
end

function FourierOperator(ch::Pair{S, S}, modes::NTuple{N, S}, σ=identity) where {S<:Integer, N}
    return Chain(
        Parallel(+, Dense(ch.first, ch.second), SpectralConv(ch, modes)),
        x -> σ.(x)
    )
end
