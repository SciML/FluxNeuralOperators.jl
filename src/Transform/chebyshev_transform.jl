export ChebyshevTransform

struct ChebyshevTransform{N, S}<:AbstractTransform
    modes::NTuple{N, S} # N == ndims(x)
end

Base.ndims(::ChebyshevTransform{N}) where {N} = N

function transform(t::ChebyshevTransform{N}, 𝐱::AbstractArray) where {N}
    return FFTW.r2r(𝐱, FFTW.REDFT00, 1:N) # [size(x)..., in_chs, batch]
end

function low_pass(t::ChebyshevTransform, 𝐱̂::AbstractArray)
    return view(𝐱̂, map(d->1:d, t.modes)..., :, :) # [ft.modes..., in_chs, batch]
end

function inverse(t::ChebyshevTransform{N}, 𝐱̂::AbstractArray) where {N}
    return FFTW.r2r(
        𝐱̂ ./ (prod(2 .* (size(𝐱̂)[1:N] .- 1))),
        FFTW.REDFT00,
        1:N,
    ) # [size(x)..., in_chs, batch]
end
