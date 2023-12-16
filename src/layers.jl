"""
    OperatorConv([rng::AbstractRNG = __default_rng()], ch::Pair{<:Integer, <:Integer},
        modes::NTuple{N, <:Integer}, ::Type{TR}; init_weight = glorot_uniform,
        T::Type{TP} = ComplexF32,
        permuted::Val{P} = Val(false)) where {N, TR <: AbstractTransform, TP, P}

## Arguments

  - `rng`: Random number generator.
  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The traform to operate the transformation.

## Keyword Arguments

  - `init_weight`: Initial function to initialize parameters.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(true)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.
  - `T`: Datatype of parameters.

## Example

```julia-repl
julia> OperatorConv(2 => 5, (16,), FourierTransform)
OperatorConv{FourierTransform}(2 => 5, (16,); permuted = false)()  # 160 parameters, plus 2 non-trainable

julia> OperatorConv(2 => 5, (16,), FourierTransform; permuted=Val(true))
OperatorConv{FourierTransform}(2 => 5, (16,); permuted = true)()  # 160 parameters, plus 2 non-trainable
```
"""
function OperatorConv(rng::AbstractRNG, ch::Pair{<:Integer, <:Integer},
        modes::NTuple{N, <:Integer}, ::Type{TR}; init_weight=glorot_uniform,
        T::Type{TP}=ComplexF32, permuted::Val=False) where {N, TR <: AbstractTransform, TP}
    in_chs, out_chs = ch
    scale = real(one(T)) / (in_chs * out_chs)
    weights = scale * init_weight(rng, T, out_chs, in_chs, prod(modes))
    transform = TR(modes)
    name = "OperatorConv{$TR}($in_chs => $out_chs, $modes; permuted = $permuted)"

    if permuted === True
        return @compact(; modes, weights, transform, dispatch=:OperatorConv,
            name) do x::AbstractArray{<:Real, M} where {M}
            y = __operator_conv(x, transform, weights)
            return y
        end
    else
        return @compact(; modes, weights, transform, dispatch=:OperatorConv,
            name) do x::AbstractArray{<:Real, M} where {M}
            N_ = ndims(transform)
            xᵀ = permutedims(x, (ntuple(i -> i + 1, N_)..., 1, N_ + 2))
            yᵀ = __operator_conv(xᵀ, transform, weights)
            y = permutedims(yᵀ, (N_ + 1, 1:N_..., N_ + 2))
            return y
        end
    end
end

"""
    SpectralConv(args...; kwargs...)

Construct a `OperatorConv` with `FourierTransform` as the transform. See
[`OperatorConv`](@ref) for the individual arguments.

## Example

```julia-repl
julia> SpectralConv(2 => 5, (16,))
OperatorConv{FourierTransform}(2 => 5, (16,); permuted = false)()  # 160 parameters, plus 2 non-trainable

julia> SpectralConv(2 => 5, (16,); permuted=Val(true))
OperatorConv{FourierTransform}(2 => 5, (16,); permuted = true)()  # 160 parameters, plus 2 non-trainable
```
"""
SpectralConv(args...; kwargs...) = OperatorConv(args..., FourierTransform; kwargs...)

"""
    OperatorKernel([rng::AbstractRNG = __default_rng()], ch::Pair{<:Integer, <:Integer},
        modes::NTuple{N, <:Integer}, transform::Type{TR}; σ = identity,
        permuted::Val{P} = Val(false), kwargs...) where {N, TR <: AbstractTransform, P}

## Arguments

  - `rng`: Random number generator.
  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The traform to operate the transformation.

## Keyword Arguments

  - `σ`: Activation function.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(true)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.
  - `T`: Datatype of parameters.

## Example

```julia-repl
julia> OperatorKernel(2 => 5, (16,), FourierTransform)
@compact(
    l₁ = Dense(2 => 5),                 # 15 parameters
    l₂ = OperatorConv{FourierTransform}(2 => 5, (16,); permuted = false)(),  # 160 parameters, plus 2
    activation = σ,
) do x::(AbstractArray{<:Real, M} where M)
    return activation.(l₁(x) .+ l₂(x))
end       # Total: 175 parameters,
          #        plus 3 states.

julia> OperatorKernel(2 => 5, (16,), FourierTransform; permuted=Val(true))
@compact(
    l₁ = Conv((1,), 2 => 5),            # 15 parameters
    l₂ = OperatorConv{FourierTransform}(2 => 5, (16,); permuted = true)(),  # 160 parameters, plus 2
    activation = σ,
) do x::(AbstractArray{<:Real, M} where M)
    return activation.(l₁(x) .+ l₂(x))
end       # Total: 175 parameters,
          #        plus 3 states.
```
"""
function OperatorKernel(rng::AbstractRNG, ch::Pair{<:Integer, <:Integer},
        modes::NTuple{N, <:Integer}, transform::Type{TR}; σ=identity,
        permuted::Val=False, kwargs...) where {N, TR <: AbstractTransform}
    l₁ = permuted === True ? Conv(map(_ -> 1, modes), ch) : Dense(ch)
    l₂ = OperatorConv(rng, ch, modes, transform; permuted, kwargs...)

    return @compact(; l₁, l₂, activation=σ,
        dispatch=:OperatorKernel) do x::AbstractArray{<:Real, M} where {M}
        return activation.(l₁(x) .+ l₂(x))
    end
end

"""
    SpectralKernel(args...; kwargs...)

Construct a `OperatorKernel` with `FourierTransform` as the transform. See
[`OperatorKernel`](@ref) for the individual arguments.

## Example

```julia-repl
julia> SpectralKernel(2 => 5, (16,))
@compact(
    l₁ = Dense(2 => 5),                 # 15 parameters
    l₂ = OperatorConv{FourierTransform}(2 => 5, (16,); permuted = false)(),  # 160 parameters, plus 2
    activation = σ,
) do x::(AbstractArray{<:Real, M} where M)
    return activation.(l₁(x) .+ l₂(x))
end       # Total: 175 parameters,
          #        plus 3 states.

julia> SpectralKernel(2 => 5, (16,); permuted=Val(true))
@compact(
    l₁ = Conv((1,), 2 => 5),            # 15 parameters
    l₂ = OperatorConv{FourierTransform}(2 => 5, (16,); permuted = true)(),  # 160 parameters, plus 2
    activation = σ,
) do x::(AbstractArray{<:Real, M} where M)
    return activation.(l₁(x) .+ l₂(x))
end       # Total: 175 parameters,
          #        plus 3 states.
```
"""
SpectralKernel(args...; kwargs...) = OperatorKernel(args..., FourierTransform; kwargs...)

# Building Blocks
function BasicBlock(rng::AbstractRNG, ch::Integer, modes::NTuple{N, <:Integer}, args...;
        add_mlp::Val=False, normalize::Val=False, σ=gelu, kwargs...) where {N}
    conv1 = SpectralConv(rng, ch => ch, modes, args...; kwargs...)
    conv2 = SpectralConv(rng, ch => ch, modes, args...; kwargs...)
    conv3 = SpectralConv(rng, ch => ch, modes, args...; kwargs...)

    kernel_size = map(_ -> 1, modes)

    if add_mlp === True
        mlp1 = Chain(Conv(kernel_size, ch => ch, σ), Conv(kernel_size, ch => ch))
        mlp2 = Chain(Conv(kernel_size, ch => ch, σ), Conv(kernel_size, ch => ch))
        mlp3 = Chain(Conv(kernel_size, ch => ch, σ), Conv(kernel_size, ch => ch))
    else
        mlp1 = NoOpLayer()
        mlp2 = NoOpLayer()
        mlp3 = NoOpLayer()
    end

    norm = normalize === True ? InstanceNorm(ch; affine = false) : NoOpLayer()

    w1 = Conv(kernel_size, ch => ch)
    w2 = Conv(kernel_size, ch => ch)
    w3 = Conv(kernel_size, ch => ch)

    return @compact(; conv1, conv2, conv3, mlp1, mlp2, mlp3, w1, w2, w3, norm, σ,
        dispatch=:BasicBlock) do (inp,)
        x, injection = __destructure(inp)

        x = norm(x)
        x1 = norm(mlp1(norm(conv1(x))))
        x2 = norm(w1(x))
        x = σ.(x1 .+ x2 .+ injection)

        x1 = norm(mlp2(norm(conv2(x))))
        x2 = norm(w2(x))
        x = σ.(x1 .+ x2 .+ injection)

        x1 = norm(mlp3(norm(conv3(x))))
        x2 = norm(w3(x))
        return σ.(x1 .+ x2 .+ injection)
    end
end

function StackedBasicBlock(rng::AbstractRNG, args...; depth::Val{N} = Val(1),
        kwargs...) where {N}
    blocks = ntuple(i -> BasicBlock(rng, args...; kwargs...), N)
    block = NamedTuple{ntuple(i -> Symbol("block_", i), N)}(blocks)

    return @compact(; block, dispatch=:StackedBasicBlock) do (inp,)
        x, injection = __destructure(inp)
        return __applyblock(block, x, injection)
    end
end

# Functional Versions
@inline function __operator_conv(x, transform, weights)
    x_t = __transform(transform, x)
    x_tr = __truncate_modes(transform, x_t)
    x_p = __apply_pattern(x_tr, weights)
    x_padded = __pad_modes(x_p, size(x_t)[1:(end - 2)]..., size(x_p)[(end - 1):end]...)
    x_inv = __inverse(transform, x_padded, size(x))
    return x_inv
end

@inline function __apply_pattern(x_tr, weights)
    x_size = size(x_tr)
    x_flat = reshape(x_tr, :, x_size[end - 1], x_size[end])
    x_weighted = __apply_pattern_batched_mul(x_flat, weights)
    return reshape(x_weighted, x_size[1:(end - 2)]..., size(x_weighted)[2:3]...)
end

@inline function __apply_pattern_batched_mul(x, y)
    # Use permutedims to guarantee contiguous memory
    x_ = permutedims(x, (2, 3, 1))         # i x b x m
    res = batched_mul(y, x_)               # o x b x m
    return permutedims(res, (3, 1, 2))     # m x o x b
end

@inline @generated function __applyblock(block::NamedTuple{names}, x, inj) where {names}
    calls = [:(x = block.$name((x, inj))) for name in names]
    return quote
        $(calls...)
    end
end

@inline __pad_modes(x, dims::Integer...) = __pad_modes(x, dims)
@inline __pad_modes(x, dims::NTuple) = __pad_modes!(similar(x, dims), x)

@inline function __pad_modes!(x_padded::AbstractArray, x::AbstractArray)
    fill!(x_padded, eltype(x)(0))
    x_padded[map(d -> 1:d, size(x))...] .= x
    return x_padded
end

@inline function CRC.rrule(::typeof(__pad_modes), x::AbstractArray, dims::NTuple)
    function ∇pad_modes(∂y)
        ∂x = view(∂y, map(Base.OneTo, size(x))...)
        return CRC.NoTangent(), ∂x, CRC.NoTangent()
    end
    return __pad_modes(x, dims), ∇pad_modes
end
