"""
	AbstractTransform

## Interface

  - `Base.ndims(<:AbstractTransform)`: N dims of modes
  - `transform(<:AbstractTransform, x::AbstractArray)`: Apply the transform to x
  - `truncate_modes(<:AbstractTransform, x_transformed::AbstractArray)`: Truncate modes
	that contribute to the noise
  - `inverse(<:AbstractTransform, x_transformed::AbstractArray)`: Apply the inverse
	transform to `x_transformed`
"""
abstract type AbstractTransform{T} end

@inline Base.eltype(::Type{<:AbstractTransform{T}}) where {T} = T

# Fourier Transform
@concrete struct FourierTransform{T} <: AbstractTransform{T}
	modes::Any
end

@inline Base.ndims(T::FourierTransform) = length(T.modes)

@inline transform(ft::FourierTransform, x::AbstractArray) = rfft(x, 1:ndims(ft))

@inline function low_pass(ft::FourierTransform, x_fft::AbstractArray)
	return view(x_fft, map(d -> 1:d, ft.modes)..., :, :)
end

@inline truncate_modes(ft::FourierTransform, x_fft::AbstractArray) = low_pass(ft, x_fft)

@inline function inverse(
	ft::FourierTransform, x_fft::AbstractArray{T, N}, M::NTuple{N, Int64}) where {T, N}
	return real(irfft(x_fft, first(M), 1:ndims(ft)))
end
