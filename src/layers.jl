"""
    OperatorConv(ch::Pair{<:Integer, <:Integer}, modes::Dims,
        ::Type{<:AbstractTransform}; init_weight=glorot_uniform,
        permuted=Val(false))

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The transform to operate the transformation.

## Keyword Arguments

  - `init_weight`: Initial function to initialize parameters.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(false)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d, batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

## Example

```jldoctest
julia> OperatorConv(2 => 5, (16,), FourierTransform{ComplexF32});

julia> OperatorConv(2 => 5, (16,), FourierTransform{ComplexF32}; permuted=Val(true));

```
"""
@concrete struct OperatorConv <: AbstractLuxLayer
    perm <: StaticBool
    in_chs::Int
    out_chs::Int
    prod_modes::Int
    tform <: AbstractTransform
    init_weight
    name::String
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::OperatorConv)
    in_chs, out_chs = layer.in_chs, layer.out_chs
    scale = real(one(eltype(layer.tform))) / (in_chs * out_chs)
    return (;
        weight=scale * layer.init_weight(
            rng, eltype(layer.tform), out_chs, in_chs, layer.prod_modes))
end

function LuxCore.parameterlength(layer::OperatorConv)
    return layer.prod_modes * layer.in_chs * layer.out_chs
end

function OperatorConv(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, ::Type{TR}; init_weight=glorot_uniform,
        permuted::BoolLike=False()) where {TR <: AbstractTransform{<:Number}}
    name = "OperatorConv{$(string(nameof(TR)))}($(ch[1]) => $(ch[2]), $modes; \
            permuted = $(dynamic(permuted)))"
    return OperatorConv(static(permuted), ch..., prod(modes), TR(modes), init_weight, name)
end

function (conv::OperatorConv{True})(x::AbstractArray, ps, st)
    return operator_conv(x, conv.tform, ps.weight), st
end

function (conv::OperatorConv{False})(x::AbstractArray, ps, st)
    N = ndims(conv.tform)
    xᵀ = permutedims(x, (ntuple(i -> i + 1, N)..., 1, N + 2))
    yᵀ = operator_conv(xᵀ, conv.tform, ps.weight)
    y = permutedims(yᵀ, (N + 1, 1:N..., N + 2))
    return y, st
end

function operator_conv(x, tform::AbstractTransform, weights)
    x_t = transform(tform, x)
    x_tr = truncate_modes(tform, x_t)
    x_p = apply_pattern(x_tr, weights)

    pad_dims = size(x_t)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = NNlib.pad_constant(x_p, expand_pad_dims(pad_dims), false;
        dims=ntuple(identity, ndims(x_p) - 2))::typeof(x_p)

    return inverse(tform, x_padded, size(x))
end

"""
    SpectralConv(args...; kwargs...)

Construct a `OperatorConv` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorConv`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralConv(2 => 5, (16,));

julia> SpectralConv(2 => 5, (16,); permuted=Val(true));

```
"""
function SpectralConv(args...; kwargs...)
    return OperatorConv(args..., FourierTransform{ComplexF32}; kwargs...)
end

"""
    OperatorKernel(ch::Pair{<:Integer, <:Integer}, modes::Dims, transform::Type{TR},
        act::A=identity; permuted=Val(false), kwargs...) where {TR <: AbstractTransform, A}

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The transform to operate the transformation.

## Keyword Arguments

  - `σ`: Activation function.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(true)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

All the keyword arguments are passed to the [`OperatorConv`](@ref) constructor.

## Example

```jldoctest
julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64});

julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64}; permuted=Val(true));

```
"""
@concrete struct OperatorKernel <: AbstractLuxWrapperLayer{:layer}
    layer
end

OperatorKernel(lin, conv) = OperatorKernel(lin, conv, identity)

function OperatorKernel(
        ch::Pair{<:Integer, <:Integer}, modes::Dims{N}, transform::Type{TR}, act=identity;
        permuted::BoolLike=False(), kwargs...) where {N, TR <: AbstractTransform{<:Number}}
    lin = known(static(permuted)) ? Conv(ntuple(one, N), ch) : Dense(ch)
    conv = OperatorConv(ch, modes, transform; permuted, kwargs...)
    return OperatorKernel(Parallel(Fix1(add_act, act), lin, conv))
end

"""
    SpectralKernel(args...; kwargs...)

Construct a `OperatorKernel` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorKernel`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralKernel(2 => 5, (16,));

julia> SpectralKernel(2 => 5, (16,); permuted=Val(true));

```
"""
function SpectralKernel(
        ch::Pair{<:Integer, <:Integer}, modes::Dims, act=identity; kwargs...)
    return OperatorKernel(ch, modes, FourierTransform{ComplexF32}, act; kwargs...)
end
