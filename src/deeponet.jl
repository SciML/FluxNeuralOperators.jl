"""
    DeepONet(; branch = (64, 32, 32, 16), trunk = (1, 8, 8, 16),
    	branch_activation = identity, trunk_activation = identity)

Constructs a DeepONet composed of Dense layers. Make sure the last node of `branch` and
`trunk` are same.

## Keyword arguments:

  - `branch`: Tuple of integers containing the number of nodes in each layer for branch net
  - `trunk`: Tuple of integers containing the number of nodes in each layer for trunk net
  - `branch_activation`: activation function for branch net
  - `trunk_activation`: activation function for trunk net
  - `additional`: `Lux` network to pass the output of DeepONet, to include additional operations
    for embeddings, defaults to `nothing`

## References

[1] Lu Lu, Pengzhan Jin, George Em Karniadakis, "DeepONet: Learning nonlinear operators for
identifying differential equations based on the universal approximation theorem of
operators", doi: https://arxiv.org/abs/1910.03193

## Example

```jldoctest
julia> deeponet = DeepONet(; branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16));

julia> ps, st = Lux.setup(Xoshiro(), deeponet);

julia> u = rand(Float32, 64, 5);

julia> y = rand(Float32, 1, 10, 5);

julia> size(first(deeponet((u, y), ps, st)))
(10, 5)
```
"""
function DeepONet(;
        branch=(64, 32, 32, 16), trunk=(1, 8, 8, 16), branch_activation=identity,
        trunk_activation=identity, additional=nothing)

    # checks for last dimension size
    @argcheck branch[end]==trunk[end] "Branch and Trunk net must share the same amount of \
                                       nodes in the last layer. Otherwise Σᵢ bᵢⱼ tᵢₖ won't \
                                       work."

    branch_net = Chain([Dense(branch[i] => branch[i + 1], branch_activation)
                        for i in 1:(length(branch) - 1)]...)

    trunk_net = Chain([Dense(trunk[i] => trunk[i + 1], trunk_activation)
                       for i in 1:(length(trunk) - 1)]...)

    return DeepONet(branch_net, trunk_net; additional)
end

"""
    DeepONet(branch, trunk)

Constructs a DeepONet from a `branch` and `trunk` architectures. Make sure that both the
nets output should have the same first dimension.

## Arguments

  - `branch`: `Lux` network to be used as branch net.
  - `trunk`: `Lux` network to be used as trunk net.

## Keyword Arguments

  - `additional`: `Lux` network to pass the output of DeepONet, to include additional operations
    for embeddings, defaults to `nothing`

## References

[1] Lu Lu, Pengzhan Jin, George Em Karniadakis, "DeepONet: Learning nonlinear operators for
identifying differential equations based on the universal approximation theorem of
operators", doi: https://arxiv.org/abs/1910.03193

## Example

```jldoctest
julia> branch_net = Chain(Dense(64 => 32), Dense(32 => 32), Dense(32 => 16));

julia> trunk_net = Chain(Dense(1 => 8), Dense(8 => 8), Dense(8 => 16));

julia> deeponet = DeepONet(branch_net, trunk_net);

julia> ps, st = Lux.setup(Xoshiro(), deeponet);

julia> u = rand(Float32, 64, 5);

julia> y = rand(Float32, 1, 10, 5);

julia> size(first(deeponet((u, y), ps, st)))
(10, 5)
```
"""
function DeepONet(branch::L1, trunk::L2; additional=nothing) where {L1, L2}
    return @compact(; branch, trunk, additional, dispatch=:DeepONet) do (u, y)
        t = trunk(y)   # p x N x nb
        b = branch(u)  # p x u_size... x nb

        @argcheck size(t, 1)==size(b, 1) "Branch and Trunk net must share the same \
                                          amount of nodes in the last layer. Otherwise \
                                          Σᵢ bᵢⱼ tᵢₖ won't work."

        @return __project(b, t, additional)
    end
end
