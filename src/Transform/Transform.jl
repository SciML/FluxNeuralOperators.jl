export
       AbstractTransform,
       transform,
       truncate_modes,
       inverse

"""
    AbstractTransform

## Interface

  - `Base.ndims(<:AbstractTransform)`: N dims of modes
  - `transform(<:AbstractTransform, 𝐱::AbstractArray)`: Apply the transform to 𝐱
  - `truncate_modes(<:AbstractTransform, 𝐱_transformed::AbstractArray)`: Truncate modes that contribute to the noise
  - `inverse(<:AbstractTransform, 𝐱_transformed::AbstractArray)`: Apply the inverse transform to 𝐱_transformed
"""
abstract type AbstractTransform end

include("utils.jl")
include("polynomials.jl")
include("fourier_transform.jl")
include("chebyshev_transform.jl")
include("wavelet_transform.jl")
