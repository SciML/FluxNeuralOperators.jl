export
       FourierNeuralOperator,
       MarkovNeuralOperator

struct FourierNeuralOperator{L, K, P} <: AbstractOperatorModel
    lifting_net::L
    integral_kernel_net::K
    project_net::P
end

Flux.@functor FourierNeuralOperator

"""
    FourierNeuralOperator(;
                          ch = (2, 64, 64, 64, 64, 64, 128, 1),
                          modes = (16, ),
                          σ = gelu)

Fourier neural operator is a operator learning model that uses Fourier kernel to perform
spectral convolutions.
It is a promising way for surrogate methods, and can be regarded as a physics operator.

The model is comprised of
a `Dense` layer to lift (d + 1)-dimensional vector field to n-dimensional vector field,
and an integral kernel operator which consists of four Fourier kernels,
and two `Dense` layers to project data back to the scalar field of interest space.

The role of each channel size described as follows:

```
[1] input channel number
 ↓ Dense
[2] lifted channel number
 ↓ OperatorKernel
[3] mapped cahnnel number
 ↓ OperatorKernel
[4] mapped cahnnel number
 ↓ OperatorKernel
[5] mapped cahnnel number
 ↓ OperatorKernel
[6] mapped cahnnel number
 ↓ Dense
[7] projected channel number
 ↓ Dense
[8] projected channel number
```

## Keyword Arguments

* `ch`: A `Tuple` or `Vector` of the 8 channel size.
* `modes`: The modes to be preserved. A tuple of length `d`,
    where `d` is the dimension of data.
* `σ`: Activation function for all layers in the model.

## Example

```julia
julia> using NNlib

julia> FourierNeuralOperator(;
                             ch = (2, 64, 64, 64, 64, 64, 128, 1),
                             modes = (16,),
                             σ = gelu)
Chain(
  Dense(2 => 64),                       # 192 parameters
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (16,), FourierTransform, permuted=false),  # 65_536 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (16,), FourierTransform, permuted=false),  # 65_536 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (16,), FourierTransform, permuted=false),  # 65_536 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (16,), FourierTransform, permuted=false),  # 65_536 parameters
    identity,
  ),
  Dense(64 => 128, gelu),               # 8_320 parameters
  Dense(128 => 1),                      # 129 parameters
)                   # Total: 18 arrays, 287_425 parameters, 2.098 MiB.
```
"""
function FourierNeuralOperator(;
                               ch = (2, 64, 64, 64, 64, 64, 128, 1),
                               modes = (16,),
                               σ = gelu)
    Transform = FourierTransform
    lifting = Dense(ch[1], ch[2])
    mapping = Chain(OperatorKernel(ch[2] => ch[3], modes, Transform, σ),
                    OperatorKernel(ch[3] => ch[4], modes, Transform, σ),
                    OperatorKernel(ch[4] => ch[5], modes, Transform, σ),
                    OperatorKernel(ch[5] => ch[6], modes, Transform))
    project = Chain(Dense(ch[6], ch[7], σ),
                    Dense(ch[7], ch[8]))

    return FourierNeuralOperator(lifting, mapping, project)
end

function (fno::FourierNeuralOperator)(𝐱::AbstractArray)
    lifted = fno.lifting_net(𝐱)
    mapped = fno.integral_kernel_net(lifted)
    𝐲 = fno.project_net(mapped)

    return 𝐲
end

struct MarkovNeuralOperator{F} <: AbstractOperatorModel
    fno::F
end

Flux.@functor MarkovNeuralOperator

"""
    MarkovNeuralOperator(;
                         ch = (1, 64, 64, 64, 64, 64, 1),
                         modes = (24, 24),
                         σ = gelu)

Markov neural operator learns a neural operator with Fourier operators.
With only one time step information of learning, it can predict the following few steps
with low loss by linking the operators into a Markov chain.

The model is comprised of
a `Dense` layer to lift d-dimensional vector field to n-dimensional vector field,
and an integral kernel operator which consists of four Fourier kernels,
and a `Dense` layers to project data back to the scalar field of interest space.

The role of each channel size described as follows:

```
[1] input channel number
 ↓ Dense
[2] lifted channel number
 ↓ OperatorKernel
[3] mapped cahnnel number
 ↓ OperatorKernel
[4] mapped cahnnel number
 ↓ OperatorKernel
[5] mapped cahnnel number
 ↓ OperatorKernel
[6] mapped cahnnel number
 ↓ Dense
[7] projected channel number
```

## Keyword Arguments

* `ch`: A `Tuple` or `Vector` of the 7 channel size.
* `modes`: The modes to be preserved. A tuple of length `d`,
    where `d` is the dimension of data.
* `σ`: Activation function for all layers in the model.

## Example

```julia
julia> using NNlib

julia> MarkovNeuralOperator(;
                            ch = (1, 64, 64, 64, 64, 64, 1),
                            modes = (24, 24),
                            σ = gelu)
Chain(
  Dense(1 => 64),                       # 128 parameters
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (24, 24), FourierTransform, permuted=false),  # 2_359_296 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (24, 24), FourierTransform, permuted=false),  # 2_359_296 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (24, 24), FourierTransform, permuted=false),  # 2_359_296 parameters
    NNlib.gelu,
  ),
  OperatorKernel(
    Dense(64 => 64),                    # 4_160 parameters
    OperatorConv(64 => 64, (24, 24), FourierTransform, permuted=false),  # 2_359_296 parameters
    NNlib.gelu,
  ),
  Dense(64 => 1),                       # 65 parameters
)                   # Total: 16 arrays, 9_454_017 parameters, 72.066 MiB.

```
"""
function MarkovNeuralOperator(;
                              ch = (1, 64, 64, 64, 64, 64, 1),
                              modes = (24, 24),
                              σ = gelu)
    Transform = FourierTransform
    lifting = Dense(ch[1], ch[2])
    mapping = Chain(OperatorKernel(ch[2] => ch[3], modes, Transform, σ),
                    OperatorKernel(ch[3] => ch[4], modes, Transform, σ),
                    OperatorKernel(ch[4] => ch[5], modes, Transform, σ),
                    OperatorKernel(ch[5] => ch[6], modes, Transform, σ))
    project = Dense(ch[6], ch[7])
    fno = FourierNeuralOperator(lifting, mapping, project)

    return MarkovNeuralOperator(fno)
end

(mno::MarkovNeuralOperator)(𝐱::AbstractArray) = mno.fno(𝐱)
