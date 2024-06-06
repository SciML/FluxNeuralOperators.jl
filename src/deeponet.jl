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
    DeepONet(rng::AbstractRNG, branch::L1, trunk::L2) where L1, L2
returns a DeepONet architecture for given branch and trunk networks
"""
function DeepONet(branch::L1, trunk::L2) where {L1, L2}

    return @compact(; branch, trunk, dispatch =:DeepONet) do (u ,  y) # ::AbstractArray{<:Real, M} where {M}
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