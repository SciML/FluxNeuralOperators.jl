function DeepONet(rng::AbstractRNG, branch::Tuple, trunk::Tuple;
    branch_activation = identity, trunk_activation = identity)\

    # checks for last dimension size

    branch_net = Chain([Dense(branch[i] => branch[i+1], branch_activation)
    for i in 1:length(branch) - 1]...);

    trunk_net = Chain([Dense(trunk[i] => trunk[i+1], trunk_activation)
    for i in 1:length(trunk) - 1]...);

    return DeepONet(rng, branch_net, trunk_net);
end
import Lux.Experimental:@compact

"""
    DeepONet(rng::AbstractRNG, branch::L1, trunk::L2) where L1, L2
returns a DeepONet architecture for given branch and trunk networks
"""
function DeepONet(rng::AbstractRNG, branch::L1, trunk::L2) where {L1, L2}

    # checks for last dimension size

    return @compact(; branch, trunk, dispatch =:DeepONet) do (u ,  y) # ::AbstractArray{<:Real, M} where {M}
        t = trunk(y); # p x N x nb 
        b = branch(u); # p x nb
        tᵀ = permutedims(t, (2, 1, 3)); # N x p x nb
        # b_ = permutedims(b, (1, 2, 3)); # p x 1 x nb
        b_ = permutedims(reshape(b, size(b)..., 1), (1, 3, 2)); # p x 1 x nb
        G = batched_mul(tᵀ, b_); # N x 1 X nb
        return dropdims(G; dims = 2);
    end
end