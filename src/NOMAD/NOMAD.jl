export NOMAD

struct NOMAD{T1, T2} <: AbstractOperatorModel
    approximator_net::T1
    decoder_net::T2
end

"""
    NOMAD(architecture_approximator::Tuple, architecture_decoder::Tuple,
          act_approximator = identity, act_decoder=true;
          init_approximator = Flux.glorot_uniform,
          init_decoder = Flux.glorot_uniform,
          bias_approximator=true, bias_decoder=true)
    NOMAD(approximator_net::Flux.Chain, decoder_net::Flux.Chain)

Create a Nonlinear Manifold Decoders for Operator Learning (NOMAD) as proposed by Lu et al.
arXiv:2206.03551

The decoder is defined as follows:

``\\tilde D (β, y) = f(β, y)``

# Usage

```julia
julia> model = NOMAD((16, 32, 16), (24, 32))
NOMAD with
Approximator net: (Chain(Dense(16 => 32), Dense(32 => 16)))
Decoder net: (Chain(Dense(24 => 32, true)))

julia> model = NeuralOperators.NOMAD((32, 64, 32), (64, 72), σ, tanh;
                                     init_approximator = Flux.glorot_normal, bias_decoder = false)
NOMAD with
Approximator net: (Chain(Dense(32 => 64, σ), Dense(64 => 32, σ)))
Decoder net: (Chain(Dense(64 => 72, tanh; bias=false)))

julia> approximator = Chain(Dense(2, 128), Dense(128, 64))
Chain(
  Dense(2 => 128),                      # 384 parameters
  Dense(128 => 64),                     # 8_256 parameters
)                   # Total: 4 arrays, 8_640 parameters, 34.000 KiB.

julia> decoder = Chain(Dense(72, 24), Dense(24, 12))
Chain(
  Dense(72 => 24),                      # 1_752 parameters
  Dense(24 => 12),                      # 300 parameters
)                   # Total: 4 arrays, 2_052 parameters, 8.266 KiB.

julia> model = NOMAD(approximator, decoder)
NOMAD with
Approximator net: (Chain(Dense(2 => 128), Dense(128 => 64)))
Decoder net: (Chain(Dense(72 => 24), Dense(24 => 12)))
```
"""
function NOMAD(architecture_approximator::Tuple, architecture_decoder::Tuple,
               act_approximator = identity, act_decoder = true;
               init_approximator = Flux.glorot_uniform,
               init_decoder = Flux.glorot_uniform,
               bias_approximator = true, bias_decoder = true)
    approximator_net = construct_subnet(architecture_approximator, act_approximator;
                                        init = init_approximator, bias = bias_approximator)

    decoder_net = construct_subnet(architecture_decoder, act_decoder;
                                   init = init_decoder, bias = bias_decoder)

    return NOMAD{typeof(approximator_net), typeof(decoder_net)}(approximator_net,
                                                                decoder_net)
end

Flux.@functor NOMAD

function (a::NOMAD)(x::AbstractArray, y::AbstractVecOrMat)
    # Assign the parameters
    approximator, decoder = a.approximator_net, a.decoder_net

    return decoder(cat(approximator(x), y', dims = 1))'
end

# Print nicely
function Base.show(io::IO, l::NOMAD)
    print(io, "NOMAD with\nApproximator net: (", l.approximator_net)
    print(io, ")\n")
    print(io, "Decoder net: (", l.decoder_net)
    print(io, ")\n")
end
