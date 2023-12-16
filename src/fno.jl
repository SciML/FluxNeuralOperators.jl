"""
    FourierNeuralOperator([rng = __default_rng()]; chs = (2, 64, 64, 64, 64, 64, 128, 1),
        modes = (16,), σ = gelu, permuted::Val = Val(false), kwargs...)

Fourier neural operator is a operator learning model that uses Fourier kernel to perform
spectral convolutions. It is a promising way for surrogate methods, and can be regarded as
a physics operator.

The model is comprised of a `Dense` layer to lift (d + 1)-dimensional vector field to
n-dimensional vector field, and an integral kernel operator which consists of four Fourier
kernels, and two `Dense` layers to project data back to the scalar field of interest space.

## Keyword Arguments

  - `chs`: A `Tuple` or `Vector` of the 8 channel size.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension
    of data.
  - `σ`: Activation function for all layers in the model.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(true)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

## Example

```julia
julia> using LuxNeuralOperators

julia> FourierNeuralOperator(; chs=(2, 64, 64, 64, 64, 64, 128, 1),
           modes=(16,), σ=gelu)
Chain(
    lifting = Dense(2 => 64),           # 192 parameters
    mapping = Chain(
        layer_1 = @compact(
            l1 = Dense(64 => 64),       # 4_160 parameters
            l2 = OperatorConv{FourierTransform}(64 => 64, (16,); permuted = false)(),  # 65_536 parameters, plus 2
            activation = σ,
        ) do x::(AbstractArray{<:Real, M} where M)
            return activation.(l1(x) .+ l2(x))
        end,
        layer_2 = @compact(
            l1 = Dense(64 => 64),       # 4_160 parameters
            l2 = OperatorConv{FourierTransform}(64 => 64, (16,); permuted = false)(),  # 65_536 parameters, plus 2
            activation = σ,
        ) do x::(AbstractArray{<:Real, M} where M)
            return activation.(l1(x) .+ l2(x))
        end,
        layer_3 = @compact(
            l1 = Dense(64 => 64),       # 4_160 parameters
            l2 = OperatorConv{FourierTransform}(64 => 64, (16,); permuted = false)(),  # 65_536 parameters, plus 2
            activation = σ,
        ) do x::(AbstractArray{<:Real, M} where M)
            return activation.(l1(x) .+ l2(x))
        end,
        layer_4 = @compact(
            l1 = Dense(64 => 64),       # 4_160 parameters
            l2 = OperatorConv{FourierTransform}(64 => 64, (16,); permuted = false)(),  # 65_536 parameters, plus 2
            activation = σ,
        ) do x::(AbstractArray{<:Real, M} where M)
            return activation.(l1(x) .+ l2(x))
        end,
    ),
    project = Chain(
        layer_1 = Dense(64 => 128, gelu),  # 8_320 parameters
        layer_2 = Dense(128 => 1),      # 129 parameters
    ),
)         # Total: 287_425 parameters,
          #        plus 12 states.
```
"""
function FourierNeuralOperator(rng::AbstractRNG; chs=(2, 64, 64, 64, 64, 64, 128, 1),
        modes=(16,), σ=gelu, permuted::Val{P}=False, kwargs...) where {P}
    @assert length(chs) ≥ 5

    map₁ = chs[1] => chs[2]
    map₂ = chs[end - 2] => chs[end - 1]
    map₃ = chs[end - 1] => chs[end]

    kernel_size = map(_ -> 1, modes)

    lifting = permuted === True ? Conv(kernel_size, map₁) : Dense(map₁)
    project = permuted === True ?
              Chain(Conv(kernel_size, map₂, σ), Conv(kernel_size, map₃)) :
              Chain(Dense(map₂, σ), Dense(map₃))

    return Chain(; lifting,
        mapping=Chain([SpectralKernel(rng, chs[i] => chs[i + 1], modes; σ, permuted,
            kwargs...) for i in 2:(length(chs) - 3)]...),
        project)
end
