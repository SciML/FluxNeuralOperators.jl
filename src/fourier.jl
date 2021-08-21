export
    SpectralConv,
    SpectralConvPerm,
    FourierOperator

abstract type AbstractSpectralConv{N, T, S, F} end

struct SpectralConv{N, T, S, F} <: AbstractSpectralConv{N, T, S, F}
    weight::T
    in_channel::S
    out_channel::S
    modes::NTuple{N, S}
    σ::F
end

struct SpectralConvPerm{N, T, S, F} <: AbstractSpectralConv{N, T, S, F}
    weight::T
    in_channel::S
    out_channel::S
    modes::NTuple{N, S}
    σ::F
end

"""
    SpectralConv(
        ch, modes, σ=identity;
        init=c_glorot_uniform, permuted=false, T=ComplexF32
    )

## SpectralConv

* ``v(x)``: input
* ``F``, ``F^{-1}``: Fourier transform, inverse fourier transform
* ``L``: linear transform on the lower Fouier modes.

``v(x)`` -> ``F`` -> ``L`` -> ``F^{-1}``

## Example

```jldoctest
julia> SpectralConv(2=>5, (16, ))
SpectralConv(2 => 5, (16,), σ=identity)

julia> SpectralConv(2=>5, (16, ), relu)
SpectralConv(2 => 5, (16,), σ=relu)

julia> SpectralConv(2=>5, (16, ), relu, permuted=true)
SpectralConvPerm(2 => 5, (16,), σ=relu)
````
"""
function SpectralConv(
    ch::Pair{S, S},
    modes::NTuple{N, S},
    σ=identity;
    init=c_glorot_uniform,
    permuted=false,
    T::DataType=ComplexF32
) where {S<:Integer, N}
    in_chs, out_chs = ch
    scale = one(T) / (in_chs * out_chs)
    weights = scale * init(out_chs, in_chs, prod(modes))

    L = permuted ? SpectralConvPerm : SpectralConv

    return L(weights, in_chs, out_chs, modes, σ)
end

Flux.@functor SpectralConv
Flux.@functor SpectralConvPerm

Base.ndims(::AbstractSpectralConv{N}) where {N} = N

function Base.show(io::IO, l::AbstractSpectralConv)
    T = (l isa SpectralConv) ? SpectralConv : SpectralConvPerm
    print(io, "$(string(T))($(l.in_channel) => $(l.out_channel), $(l.modes), σ=$(string(l.σ)))")
end

function spectral_conv(m::AbstractSpectralConv, 𝐱::AbstractArray)
    n_dims = ndims(𝐱)

    𝐱_fft = fft(Zygote.hook(real, 𝐱), 1:ndims(m)) # [x, in_chs, batch]
    𝐱_flattened = reshape(view(𝐱_fft, map(d->1:d, m.modes)..., :, :), :, size(𝐱_fft, n_dims-1), size(𝐱_fft, n_dims))
    𝐱_weighted = apply_spectral_pattern(𝐱_flattened, m.weight) # [prod(m.modes), out_chs, batch], only 3-dims
    𝐱_shaped = reshape(𝐱_weighted, m.modes..., size(𝐱_weighted, 2), size(𝐱_weighted, 3))
    𝐱_padded = spectral_pad(𝐱_shaped, (size(𝐱_fft)[1:end-2]..., size(𝐱_weighted, 2), size(𝐱_weighted, 3))) # [x, out_chs, batch] <- [modes, out_chs, batch]
    𝐱_ifft = real(ifft(𝐱_padded, 1:ndims(m))) # [x, out_chs, batch]

    return m.σ.(𝐱_ifft)
end

function (m::SpectralConv)(𝐱)
    𝐱ᵀ = permutedims(𝐱, (ntuple(i->i+1, ndims(m))..., 1, ndims(m)+2)) # [x, in_chs, batch] <- [in_chs, x, batch]
    𝐱_out = spectral_conv(m, 𝐱ᵀ) # [x, out_chs, batch]
    𝐱_outᵀ = permutedims(𝐱_out, (ndims(m)+1, 1:ndims(m)..., ndims(m)+2)) # [out_chs, x, batch] <- [x, out_chs, batch]

    return 𝐱_outᵀ
end

function (m::SpectralConvPerm)(𝐱)
    return spectral_conv(m, 𝐱) # [x, out_chs, batch]
end

############
# operator #
############

function FourierOperator(
    ch::Pair{S, S},
    modes::NTuple{N, S},
    σ=identity;
    permuted=false
) where {S<:Integer, N}
    short_cut = permuted ? Conv(Tuple(ones(Int, length(modes))), ch) : Dense(ch.first, ch.second)

    return Chain(
        Parallel(+, short_cut, SpectralConv(ch, modes, permuted=permuted)),
        x -> σ.(x)
    )
end

#########
# utils #
#########

c_glorot_uniform(dims...) = Flux.glorot_uniform(dims...) + Flux.glorot_uniform(dims...)*im

# [prod(modes), out_chs, batch] <- [prod(modes), in_chs, batch] * [out_chs, in_chs, prod(modes)]
apply_spectral_pattern(𝐱₁, 𝐱₂) = @tullio 𝐲[m, o, b] := 𝐱₁[m, i, b] * 𝐱₂[o, i, m]

spectral_pad(𝐱::AbstractArray, dims::NTuple) = spectral_pad!(similar(𝐱, dims), 𝐱)

function spectral_pad!(𝐱_padded::AbstractArray, 𝐱::AbstractArray)
    fill!(𝐱_padded, eltype(𝐱)(0)) # zeros(eltype(𝐱), dims)
    𝐱_padded[map(d->1:d, size(𝐱))...] .= 𝐱

    return 𝐱_padded
end

function ChainRulesCore.rrule(::typeof(spectral_pad), 𝐱::AbstractArray, dims::NTuple)
    function spectral_pad_pullback(𝐲̄)
        return NoTangent(), view(𝐲̄, map(d->1:d, size(𝐱))...), NoTangent()
    end

    return spectral_pad(𝐱, dims), spectral_pad_pullback
end
