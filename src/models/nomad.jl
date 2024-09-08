"""
    NOMAD(approximator, decoder, concatenate)

Constructs a NOMAD from `approximator` and `decoder` architectures. Make sure the output
from `approximator` combined with the coordinate dimension has compatible size for input to
`decoder`

## Arguments

  - `approximator`: `Lux` network to be used as approximator net.
  - `decoder`: `Lux` network to be used as decoder net.

## Keyword Arguments

  - `concatenate`: function that defines the concatenation of output from `approximator` and
    the coordinate dimension, defaults to concatenation along first dimension after
    vectorizing the tensors

## References

[1] Jacob H. Seidman and Georgios Kissas and Paris Perdikaris and George J. Pappas, "NOMAD:
Nonlinear Manifold Decoders for Operator Learning", doi: https://arxiv.org/abs/2206.03551

## Example

```jldoctest
julia> approximator_net = Chain(Dense(8 => 32), Dense(32 => 32), Dense(32 => 16));

julia> decoder_net = Chain(Dense(18 => 16), Dense(16 => 16), Dense(16 => 8));

julia> nomad = NOMAD(approximator_net, decoder_net);

julia> ps, st = Lux.setup(Xoshiro(), nomad);

julia> u = rand(Float32, 8, 5);

julia> y = rand(Float32, 2, 5);

julia> size(first(nomad((u, y), ps, st)))
(8, 5)
```
"""
@concrete struct NOMAD <: AbstractLuxWrapperLayer{:model}
    model <: Chain
end

"""
    NOMAD(; approximator = (8, 32, 32, 16), decoder = (18, 16, 8, 8),
        approximator_activation = identity, decoder_activation = identity)

Constructs a NOMAD composed of Dense layers. Make sure that last node of `approximator` +
coordinate length = first node of `decoder`.

## Keyword arguments:

  - `approximator`: Tuple of integers containing the number of nodes in each layer for
    approximator net
  - `decoder`: Tuple of integers containing the number of nodes in each layer for decoder
    net
  - `approximator_activation`: activation function for approximator net
  - `decoder_activation`: activation function for decoder net
  - `concatenate`: function that defines the concatenation of output from `approximator` and
    the coordinate dimension, defaults to concatenation along first dimension after
    vectorizing the tensors

## References

[1] Jacob H. Seidman and Georgios Kissas and Paris Perdikaris and George J. Pappas, "NOMAD:
Nonlinear Manifold Decoders for Operator Learning", doi: https://arxiv.org/abs/2206.03551

## Example

```jldoctest
julia> nomad = NOMAD(; approximator=(8, 32, 32, 16), decoder=(18, 16, 8, 8));

julia> ps, st = Lux.setup(Xoshiro(), nomad);

julia> u = rand(Float32, 8, 5);

julia> y = rand(Float32, 2, 5);

julia> size(first(nomad((u, y), ps, st)))
(8, 5)
```
"""
function NOMAD(; approximator=(8, 32, 32, 16), decoder=(18, 16, 8, 8),
        approximator_activation=identity,
        decoder_activation=identity, concatenate=nomad_concatenate)
    approximator_net = Chain([Dense(approximator[i] => approximator[i + 1],
                                  approximator_activation)
                              for i in 1:(length(approximator) - 1)]...)

    decoder_net = Chain([Dense(decoder[i] => decoder[i + 1], decoder_activation)
                         for i in 1:(length(decoder) - 1)]...)

    return NOMAD(approximator_net, decoder_net, concatenate)
end

function NOMAD(approximator, decoder, concatenate=nomad_concatenate)
    return NOMAD(Chain(Parallel(concatenate, approximator, NoOpLayer()), decoder))
end

batch_vectorize(x::AbstractArray) = reshape(x, :, size(x, ndims(x)))

nomad_concatenate(x::AbstractMatrix, y::AbstractMatrix) = cat(x, y; dims=1)
function nomad_concatenate(x::AbstractArray, y::AbstractArray)
    return nomad_concatenate(batch_vectorize(x), batch_vectorize(y))
end
