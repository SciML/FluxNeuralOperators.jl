export
    SpectralConv,
    FourierOperator

struct SpectralConv{N, T, S}
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
    weights = scale * init(out_chs, in_chs, prod(modes))

    return SpectralConv(weights, in_chs, out_chs, modes, N, σ)
end

Flux.@functor SpectralConv

Base.ndims(::SpectralConv{N}) where {N} = N

# [prod(m.modes), out_chs, batch] <- [prod(m.modes), in_chs, batch] * [out_chs, in_chs, prod(m.modes)]
spectral_conv(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[o, i, m]

function (m::SpectralConv)(𝐱::AbstractArray)
    n_dims = ndims(𝐱)

    𝐱ᵀ = permutedims(Zygote.hook(real, 𝐱), (ntuple(i->i+1, ndims(m))..., 1, ndims(m)+2)) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_fft = fft(𝐱ᵀ, 1:ndims(m)) # [x, in_chs, batch]

    𝐱_flattened = reshape(view(𝐱_fft, map(d->1:d, m.modes)..., :, :), :, size(𝐱_fft, n_dims-1), size(𝐱_fft, n_dims))
    𝐱_weighted = spectral_conv(𝐱_flattened, m.weight) # [prod(m.modes), out_chs, batch], only 3-dims
    𝐱_shaped = reshape(𝐱_weighted, m.modes..., size(𝐱_weighted, 2), size(𝐱_weighted, 3))

    # [x, out_chs, batch] <- [modes, out_chs, batch]
    pad = zeros(ComplexF32, ntuple(i->size(𝐱_fft, i)-m.modes[i], ndims(m))..., size(𝐱_shaped, n_dims-1), size(𝐱_shaped, n_dims))
    𝐱_padded = cat(𝐱_shaped, pad, dims=1:ndims(m))

    𝐱_out = ifft(𝐱_padded, 1:ndims(m)) # [x, out_chs, batch]
    𝐱_outᵀ = permutedims(real(𝐱_out), (ndims(m)+1, 1:ndims(m)..., ndims(m)+2)) # [out_chs, x, batch] <- [x, out_chs, batch]

    return m.σ.(𝐱_outᵀ)
end

function FourierOperator(ch::Pair{S, S}, modes::NTuple{N, S}, σ=identity) where {S<:Integer, N}
    return Chain(
        Parallel(+, Dense(ch.first, ch.second), SpectralConv(ch, modes)),
        x -> σ.(x)
    )
end
