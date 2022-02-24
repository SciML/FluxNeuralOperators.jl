"""
Construct a Chain of `Dense` layers from a given tuple of integers.

Input:
A tuple (m,n,o,p) of integer type numbers that each describe the width of the i-th Dense layer to Construct

Output:
A `Flux` Chain with length of the input tuple and individual width given by the tuple elements

# Example

```julia
julia> model = NeuralOperators.construct_subnet((2,128,64,32,1))
Chain(
  Dense(2, 128),                        # 384 parameters
  Dense(128, 64),                       # 8_256 parameters
  Dense(64, 32),                        # 2_080 parameters
  Dense(32, 1),                         # 33 parameters
)                   # Total: 8 arrays, 10_753 parameters, 42.504 KiB.

julia> model([2,1])
1-element Vector{Float32}:
 -0.7630446
```
"""
function construct_subnet(architecture::Tuple, σ = identity;
                          init=Flux.glorot_uniform, bias=true)
    # First, create an array that contains all Dense layers independently
    # Given n-element architecture constructs n-1 layers
    layers = Array{Flux.Dense}(undef, length(architecture)-1)
    @inbounds for i ∈ 2:length(architecture)
      layers[i-1] = Flux.Dense(architecture[i-1], architecture[i], σ;
                                init=init, bias=bias)
    end

    # Concatenate the layers to a string, chain them and parse them into
    # the Flux Chain constructor syntax
    return Meta.parse("Chain("*join(layers,",")*")") |> eval
end
