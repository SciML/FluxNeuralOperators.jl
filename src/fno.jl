"""
    FourierNeuralOperator(
        σ=gelu; chs::Dims{C}=(2, 64, 64, 64, 64, 64, 128, 1), modes::Dims{M}=(16,),
        permuted::Val{perm}=False, kwargs...) where {C, M, perm}

Fourier neural operator is a operator learning model that uses Fourier kernel to perform
spectral convolutions. It is a promising way for surrogate methods, and can be regarded as
a physics operator.

The model is comprised of a `Dense` layer to lift (d + 1)-dimensional vector field to
n-dimensional vector field, and an integral kernel operator which consists of four Fourier
kernels, and two `Dense` layers to project data back to the scalar field of interest space.

## Arguments

  - `σ`: Activation function for all layers in the model.

## Keyword Arguments

  - `chs`: A `Tuple` or `Vector` of the 8 channel size.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension
    of data.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(false)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

## Example

```jldoctest
julia> FourierNeuralOperator(gelu; chs=(2, 64, 64, 128, 1), modes=(16,))
FourierNeuralOperator(
    lifting = Dense(2 => 64),           # 192 parameters
    mapping = @compact(
        l₁ = Dense(64 => 64),           # 4_160 parameters
        l₂ = OperatorConv{FourierTransform}(64 => 64, (16,); permuted = false)(),  # 65_536 parameters
        activation = gelu,
    ) do x::AbstractArray
        l₁x = l₁(x)
        l₂x = l₂(x)
        return @__dot__(activation(l₁x + l₂x))
    end,
    project = Chain(
        layer_1 = Dense(64 => 128, gelu),  # 8_320 parameters
        layer_2 = Dense(128 => 1),      # 129 parameters
    ),
)         # Total: 78_337 parameters,
          #        plus 1 states.
```
"""
function FourierNeuralOperator(
        σ=gelu; chs::Dims{C}=(2, 64, 64, 64, 64, 64, 128, 1), modes::Dims{M}=(16,),
        permuted::Val{perm}=Val(false), kwargs...) where {C, M, perm}
    @argcheck length(chs) ≥ 5

    map₁ = chs[1] => chs[2]
    map₂ = chs[C - 2] => chs[C - 1]
    map₃ = chs[C - 1] => chs[C]

    kernel_size = map(Returns(1), modes)

    lifting = perm ? Conv(kernel_size, map₁) : Dense(map₁)
    project = perm ? Chain(Conv(kernel_size, map₂, σ), Conv(kernel_size, map₃)) :
              Chain(Dense(map₂, σ), Dense(map₃))

    return Chain(; lifting,
        mapping=Chain([SpectralKernel(chs[i] => chs[i + 1], modes, σ; permuted, kwargs...)
                       for i in 2:(C - 3)]...),
        project,
        name="FourierNeuralOperator")
end
