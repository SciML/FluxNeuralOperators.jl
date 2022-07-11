export WaveletTransform

struct WaveletTransform{N, S} <: AbstractTransform
    ec_d::Any
    ec_s::Any
    modes::NTuple{N, S} # N == ndims(x)
end

Base.ndims(::WaveletTransform{N}) where {N} = N

function transform(wt::WaveletTransform, 𝐱::AbstractArray)
    N = size(X, ndims(wt) - 1)
    # 1d
    Xa = vcat(view(𝐱, :, :, 1:2:N, :), view(𝐱, :, :, 2:2:N, :))
    # 2d
    # Xa = vcat(
    #     view(𝐱, :, :, 1:2:N, 1:2:N, :),
    #     view(𝐱, :, :, 1:2:N, 2:2:N, :),
    #     view(𝐱, :, :, 2:2:N, 1:2:N, :),
    #     view(𝐱, :, :, 2:2:N, 2:2:N, :),
    # )
    d = NNlib.batched_mul(Xa, wt.ec_d)
    s = NNlib.batched_mul(Xa, wt.ec_s)
    return d, s
end

function inverse(wt::WaveletTransform, 𝐱_fwt::AbstractArray) end

# function truncate_modes(wt::WaveletTransform, 𝐱_fft::AbstractArray)
#     return view(𝐱_fft, map(d->1:d, wt.modes)..., :, :) # [ft.modes..., in_chs, batch]
# end
