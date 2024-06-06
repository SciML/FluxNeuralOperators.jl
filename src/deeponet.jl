"""
    DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16),
        branch_activation = identity, trunk_activation = identity)

constructs a DeepONet composed of Dense layers. Make sure the last node of `branch` and `trunk` are same.

## Keyword arguments:

- `branch`: Tuple of integers containing the number of nodes in each layer for branch net
- `trunk`: Tuple of integers containing the number of nodes in each layer for trunk net
- `branch_activation`: activation function for branch net
- `trunk_activation`: activation function for trunk net

## References:
Lu Lu, Pengzhan Jin, George Em Karniadakis , "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators"
doi: https://arxiv.org/abs/1910.03193

## Example:
```julia-repl
julia> deeponet = DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16))

Branch : Chain(
    layer_1 = Dense(64 => 32),          # 2_080 parameters
    layer_2 = Dense(32 => 32),          # 1_056 parameters
    layer_3 = Dense(32 => 16),          # 528 parameters
)         # Total: 3_664 parameters,
          #        plus 0 states.
 
  Trunk: Chain(
    layer_1 = Dense(1 => 8),            # 16 parameters
    layer_2 = Dense(8 => 8),            # 72 parameters
    layer_3 = Dense(8 => 16),           # 144 parameters
)         # Total: 232 parameters,
          #        plus 0 states.
()  # 3_896 parameters
```
"""
function DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16),
    branch_activation = identity, trunk_activation = identity)

    # checks for last dimension size
    @assert branch[end] == trunk[end] "Branch and Trunk net must share the same amount of nodes in the last layer. 
    Otherwise Σᵢ bᵢⱼ tᵢₖ won't work."

    branch_net = Chain([Dense(branch[i] => branch[i+1], branch_activation)
    for i in 1:length(branch) - 1]...);

    trunk_net = Chain([Dense(trunk[i] => trunk[i+1], trunk_activation)
    for i in 1:length(trunk) - 1]...);

    return DeepONet(branch_net, trunk_net);
end

"""
    DeepONet(branch::L1, trunk::L2) where {L1, L2}
constructs a DeepONet from a `branch` and `trunk` architectures. 
Make sure that both the nets output should have the same first dimension.

## Arguments:
- `branch`: `Lux` network to be used as branch net. 
- `trunk`: `Lux` network to be used as trunk net. 


## References:
Lu Lu, Pengzhan Jin, George Em Karniadakis , "DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators"
doi: https://arxiv.org/abs/1910.03193

## Example:

```julia-repl
julia> branch_net =  Chain(Dense(64 =>32), Dense(32 =>32), Dense(32 => 16));
julia> trunk_net = Chain(Dense(1 =>8), Dense(8 =>8), Dense(8 => 16));
julia> don_ = DeepONet(branch_net, trunk_net)

Branch : Chain(
    layer_1 = Dense(64 => 32),          # 2_080 parameters        
    layer_2 = Dense(32 => 32),          # 1_056 parameters        
    layer_3 = Dense(32 => 16),          # 528 parameters
)         # Total: 3_664 parameters,
          #        plus 0 states.

  Trunk: Chain(
    layer_1 = Dense(1 => 8),            # 16 parameters
    layer_2 = Dense(8 => 8),            # 72 parameters
    layer_3 = Dense(8 => 16),           # 144 parameters
)         # Total: 232 parameters,
          #        plus 0 states.
()  # 3_896 parameters
```
"""
function DeepONet(branch::L1, trunk::L2) where {L1, L2}

    io = IOBuffer();
    td= TextDisplay(io);
    display(td, branch);
    branch_name = String(take!(io));

    display(td, trunk);
    trunk_name = String(take!(io));

    name = "Branch : $branch_name \n  Trunk: $trunk_name"

    return @compact(; branch, trunk, dispatch =:DeepONet, name) do (u ,  y) # ::AbstractArray{<:Real, M} where {M}
        t = trunk(y); # p x N x nb 
        b = branch(u); # p x nb

        # checks for last dimension size
        @assert size(t, 1)== size(b, 1) "Branch and Trunk net must share the same amount of nodes in the last layer. 
        Otherwise Σᵢ bᵢⱼ tᵢₖ won't work."

        tᵀ = permutedims(t, (2, 1, 3)); # N x p x nb
        b_ = permutedims(reshape(b, size(b)..., 1), (1, 3, 2)); # p x 1 x nb
        G = batched_mul(tᵀ, b_); # N x 1 X nb
        return dropdims(G; dims = 2);
    end
end