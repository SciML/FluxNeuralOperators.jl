"""
    AbstractTransform

## Interface

  - `Base.ndims(<:AbstractTransform)`: N dims of modes
  - `__transform(<:AbstractTransform, x::AbstractArray)`: Apply the transform to x
  - `__truncate_modes(<:AbstractTransform, x_transformed::AbstractArray)`: Truncate modes
    that contribute to the noise
  - `__inverse(<:AbstractTransform, x_transformed::AbstractArray)`: Apply the inverse
    transform to `x_transformed`
"""
abstract type AbstractTransform end

# Fourier Transform
struct FourierTransform{N, S} <: AbstractTransform
    modes::NTuple{N, S}
end

Base.ndims(::FourierTransform{N}) where {N} = N
Base.eltype(::Type{FourierTransform}) = ComplexF32

@inline __transform(ft::FourierTransform, x::AbstractArray) = rfft(x, 1:ndims(ft))

@inline function __low_pass(ft::FourierTransform, x_fft::AbstractArray)
    return view(x_fft, map(d -> 1:d, ft.modes)..., :, :)
end

@inline __truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = __low_pass(ft, x_fft)

function __inverse(ft::FourierTransform, x_fft::AbstractArray{T, N},
        M::NTuple{N, Int64}) where {T, N}
    return real(irfft(x_fft, first(M), 1:ndims(ft)))
end
