"""
    DeepONet(branch, trunk, additional)

Constructs a DeepONet from a `branch` and `trunk` architectures. Make sure that both the
nets output should have the same first dimension.

## Arguments

  - `branch`: `Lux` network to be used as branch net.
  - `trunk`: `Lux` network to be used as trunk net.

## Keyword Arguments

  - `additional`: `Lux` network to pass the output of DeepONet, to include additional
    operations for embeddings, defaults to `nothing`

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
@concrete struct DeepONet <: AbstractLuxContainerLayer{(:branch, :trunk, :additional)}
    branch
    trunk
    additional
end

DeepONet(branch, trunk) = DeepONet(branch, trunk, NoOpLayer())

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
  - `additional`: `Lux` network to pass the output of DeepONet, to include additional
    operations for embeddings, defaults to `nothing`

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
        trunk_activation=identity, additional=NoOpLayer())

    # checks for last dimension size
    @argcheck branch[end]==trunk[end] "Branch and Trunk net must share the same amount of \
                                       nodes in the last layer. Otherwise Σᵢ bᵢⱼ tᵢₖ won't \
                                       work."

    branch_net = Chain([Dense(branch[i] => branch[i + 1], branch_activation)
                        for i in 1:(length(branch) - 1)]...)

    trunk_net = Chain([Dense(trunk[i] => trunk[i + 1], trunk_activation)
                       for i in 1:(length(trunk) - 1)]...)

    return DeepONet(branch_net, trunk_net, additional)
end

function (deeponet::DeepONet)((x1, x2), ps, st::NamedTuple)
    b, st_b = deeponet.branch(x1, ps.branch, st.branch)
    t, st_t = deeponet.trunk(x2, ps.trunk, st.trunk)

    @argcheck size(b, 1)==size(t, 1) "Branch and Trunk net must share the same amount of \
                                      nodes in the last layer. Otherwise Σᵢ bᵢⱼ tᵢₖ won't \
                                      work."

    additional = deeponet.additional isa NoOpLayer ? nothing :
                 StatefulLuxLayer{true}(deeponet.additional, ps.additional, st.additional)
    out = deeponet_project(b, t, additional)

    stₙ = merge((; branch=st_b, trunk=st_t),
        deeponet.additional isa NoOpLayer ? (;) : additional.st)
    return out, stₙ
end

function deeponet_project(
        b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}, ::Nothing) where {T1, T2}
    # b [p, nb], t [p, N, nb]
    bᵣ = reshape(b, size(b, 1), 1, size(b, 2))
    return dropdims(sum(bᵣ .* t; dims=1); dims=1) # [N, nb]
end

function deeponet_project(
        b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}, ::Nothing) where {T1, T2}
    # b [p, u, nb], t [p, N, nb]
    return batched_matmul(batched_adjoint(b), t) # [u, N, b]
end

function deeponet_project(
        b::AbstractArray{T1, N}, t::AbstractArray{T2, 3}, ::Nothing) where {T1, T2, N}
    # b [p, u_size..., nb], t [p, N, nb]
    bᵣ = reshape(b, size(b, 1), :, size(b, N))
    return reshape(batched_matmul(batched_adjoint(bᵣ), t),
        size(b)[2:(N - 1)]..., size(t, 2), size(b, N))
end

function deeponet_project(
        b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}, additional) where {T1, T2}
    # b [p, nb], t [p, N, nb]
    bᵣ = reshape(b, size(b, 1), 1, size(b, 2))
    return additional(bᵣ .* t) # [p, N, nb] => [out_dims, N, nb]
end

function deeponet_project(
        b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}, additional) where {T1, T2}
    # b [p, u, nb], t [p, N, nb]
    bᵣ = reshape(b, size(b, 1), size(b, 2), 1, size(b, 3)) # [p, u, 1, nb]
    tᵣ = reshape(t, size(t, 1), 1, size(t)[2:end]...)      # [p, 1, N, nb]
    return additional(bᵣ .* tᵣ) # [p, u, N, nb] => [out_size, u, N, nb]
end

function deeponet_project(
        b::AbstractArray{T1, N}, t::AbstractArray{T2, 3}, additional) where {T1, T2, N}
    # b [p, u_size..., nb], t [p, N, nb]
    bᵣ = reshape(b, size(b, 1), :, 1, size(b, N))          # [p, (u_size...), 1, nb]
    tᵣ = reshape(t, size(t, 1), 1, size(t, 2), size(t, 3)) # [p, 1, N, nb]
    bᵣtᵣ = reshape(bᵣ .* tᵣ, size(b, 1), size(b)[2:(N - 1)]..., size(t, 2), size(b, N))
    return additional(bᵣtᵣ) # [p, u_size..., N, nb] => [out_size, u_size..., N, nb]
end
