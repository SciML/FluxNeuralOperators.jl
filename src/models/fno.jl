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
julia> fno = FourierNeuralOperator(gelu; chs=(2, 64, 64, 128, 1), modes=(16,));

julia> ps, st = Lux.setup(Xoshiro(), fno);

julia> u = rand(Float32, 2, 1024, 5);

julia> size(first(fno(u, ps, st)))
(1, 1024, 5)
```
"""
@concrete struct FourierNeuralOperator <: AbstractLuxWrapperLayer{:model}
    model <: Chain
end

function FourierNeuralOperator(σ=gelu; chs::Dims{C}=(2, 64, 64, 64, 64, 64, 128, 1),
        modes::Dims{M}=(16,), permuted::BoolLike=False(), kwargs...) where {C, M}
    @argcheck length(chs) ≥ 5

    map₁ = chs[1] => chs[2]
    map₂ = chs[C - 2] => chs[C - 1]
    map₃ = chs[C - 1] => chs[C]

    kernel_size = map(Returns(1), modes)

    lifting = known(static(permuted)) ? Conv(kernel_size, map₁) : Dense(map₁)
    project = known(static(permuted)) ?
              Chain(Conv(kernel_size, map₂, σ), Conv(kernel_size, map₃)) :
              Chain(Dense(map₂, σ), Dense(map₃))

    mapping = Chain([SpectralKernel(chs[i] => chs[i + 1], modes, σ; permuted, kwargs...)
                     for i in 2:(C - 3)]...)

    return FourierNeuralOperator(Chain(lifting, mapping, project))
end
