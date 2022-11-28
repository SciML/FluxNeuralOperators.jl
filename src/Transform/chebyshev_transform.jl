export ChebyshevTransform

struct ChebyshevTransform{N, S} <: AbstractTransform
    modes::NTuple{N, S} # N == ndims(x)
end

Base.ndims(::ChebyshevTransform{N}) where {N} = N

function transform(t::ChebyshevTransform{N}, 𝐱::AbstractArray) where {N}
    return FFTW.r2r(𝐱, FFTW.REDFT10, 1:N) # [size(x)..., in_chs, batch]
end

function truncate_modes(t::ChebyshevTransform, 𝐱̂::AbstractArray)
    return view(𝐱̂, map(d -> 1:d, t.modes)..., :, :) # [t.modes..., in_chs, batch]
end

function inverse(t::ChebyshevTransform{N}, 𝐱̂::AbstractArray, M::NTuple{N, Int64}) where {N}
    normalized_𝐱̂ = 𝐱̂ ./ (prod(2 .* (size(𝐱̂)[1:N] .- 1)))
    return FFTW.r2r(normalized_𝐱̂, FFTW.REDFT01, 1:N) # [size(x)..., in_chs, batch]
end

function ChainRulesCore.rrule(::typeof(FFTW.r2r), x::AbstractArray, kind, dims)
    y = FFTW.r2r(x, kind, dims)
    r2r_pullback(Δ) = (NoTangent(), ∇r2r(unthunk(Δ), kind, dims), NoTangent(), NoTangent())
    return y, r2r_pullback
end

function ∇r2r(Δ::AbstractArray{T}, kind, dims) where {T}
    # derivative of r2r turns out to be r2r
    Δx = FFTW.r2r(Δ, kind, dims)

    # rank 4 correction: needs @bischtob to elaborate the reason using this. 
    # (M,) = size(Δ)[dims]
    # a1 = fill!(similar(Δ, M), one(T))
    # CUDA.@allowscalar a1[1] = a1[end] = zero(T)

    # a2 = fill!(similar(Δ, M), one(T))
    # a2[1:2:end] .= -one(T)
    # CUDA.@allowscalar a2[1] = a2[end] = zero(T)

    # e1 = fill!(similar(Δ, M), zero(T))
    # CUDA.@allowscalar e1[1] = one(T)

    # eN = fill!(similar(Δ, M), zero(T))
    # CUDA.@allowscalar eN[end] = one(T)

    # Δx .+= @. a1' * sum(e1' .* Δ, dims=2) - a2' * sum(eN' .* Δ, dims=2)
    # Δx .+= @. eN' * sum(a2' .* Δ, dims=2) - e1' * sum(a1' .* Δ, dims=2)
    return Δx
end
