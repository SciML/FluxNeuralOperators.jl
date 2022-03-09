"""
`DeepONet(architecture_branch::Tuple, architecture_trunk::Tuple,
                        act_branch = identity, act_trunk = identity;
                        init_branch = Flux.glorot_uniform,
                        init_trunk = Flux.glorot_uniform,
                        bias_branch=true, bias_trunk=true)`
`DeepONet(branch_net::Flux.Chain, trunk_net::Flux.Chain)`

Create an (unstacked) DeepONet architecture as proposed by Lu et al.
arXiv:1910.03193

The model works as follows:

x --- branch --
               |
                -⊠--u-
               |
y --- trunk ---

Where `x` represents the input function, discretely evaluated at its respective sensors.
So the ipnut is of shape [m] for one instance or [m x b] for a training set.
`y` are the probing locations for the operator to be trained. It has shape [N x n] for
N different variables in the PDE (i.e. spatial and temporal coordinates) with each n distinct evaluation points.
`u` is the solution of the queried instance of the PDE, given by the specific choice of parameters.

Both inputs `x` and `y` are multiplied together via dot product Σᵢ bᵢⱼ tᵢₖ.

You can set up this architecture in two ways:

1. By Specifying the architecture and all its parameters as given above. This always creates
 `Dense` layers for the branch and trunk net and corresponds to the DeepONet proposed by Lu et al.

2. By passing two architectures in the form of two Chain structs directly. Do this if you want more
flexibility and e.g. use an RNN or CNN instead of simple `Dense` layers.

Strictly speaking, DeepONet does not imply either of the branch or trunk net to be a simple
 DNN. Usually though, this is the case which is why it's treated as the default case here.

# Example

Consider a transient 1D advection problem ∂ₜu + u ⋅ ∇u = 0, with an IC u(x,0) = g(x).
We are given several (b = 200) instances of the IC, discretized at 50 points each and want
 to query the solution for 100 different locations and times [0;1].

That makes the branch input of shape [50 x 200] and the trunk input of shape [2 x 100]. So the
 input for the branch net is 50 and 100 for the trunk net.

# Usage

```julia
julia> model = DeepONet((32,64,72), (24,64,72))
DeepONet with
branch net: (Chain(Dense(32, 64), Dense(64, 72)))
Trunk net: (Chain(Dense(24, 64), Dense(64, 72)))

julia> model = DeepONet((32,64,72), (24,64,72), σ, tanh; init_branch=Flux.glorot_normal, bias_trunk=false)
DeepONet with
branch net: (Chain(Dense(32, 64, σ), Dense(64, 72, σ)))
Trunk net: (Chain(Dense(24, 64, tanh; bias=false), Dense(64, 72, tanh; bias=false)))

julia> branch = Chain(Dense(2,128),Dense(128,64),Dense(64,72))
Chain(
  Dense(2, 128),                        # 384 parameters
  Dense(128, 64),                       # 8_256 parameters
  Dense(64, 72),                        # 4_680 parameters
)                   # Total: 6 arrays, 13_320 parameters, 52.406 KiB.

julia> trunk = Chain(Dense(1,24),Dense(24,72))
Chain(
  Dense(1, 24),                         # 48 parameters
  Dense(24, 72),                        # 1_800 parameters
)                   # Total: 4 arrays, 1_848 parameters, 7.469 KiB.

julia> model = DeepONet(branch,trunk)
DeepONet with
branch net: (Chain(Dense(2, 128), Dense(128, 64), Dense(64, 72)))
Trunk net: (Chain(Dense(1, 24), Dense(24, 72)))
```
"""
struct DeepONet{T1, T2}
    branch_net::T1
    trunk_net::T2
end

# Declare the function that assigns Weights and biases to the layer
function DeepONet(architecture_branch::Tuple, architecture_trunk::Tuple,
                        act_branch = identity, act_trunk = identity;
                        init_branch = Flux.glorot_uniform,
                        init_trunk = Flux.glorot_uniform,
                        bias_branch=true, bias_trunk=true)

    @assert architecture_branch[end] == architecture_trunk[end] "Branch and Trunk net must share the same amount of nodes in the last layer. Otherwise Σᵢ bᵢⱼ tᵢₖ won't work."

    # To construct the subnets we use the helper function in subnets.jl
    # Initialize the branch net
    branch_net = construct_subnet(architecture_branch, act_branch;
                                    init=init_branch, bias=bias_branch)
    # Initialize the trunk net
    trunk_net = construct_subnet(architecture_trunk, act_trunk;
                                    init=init_trunk, bias=bias_trunk)

    return DeepONet{typeof(branch_net),typeof(trunk_net)}(branch_net, trunk_net)
end

Flux.@functor DeepONet

#= The actual layer that does stuff
x is the input function, evaluated at m locations (or m x b in case of batches)
y is the array of sensors, i.e. the variables of the output function
with shape (N x n) - N different variables with each n evaluation points =#
function (a::DeepONet)(x::AbstractArray, y::AbstractVecOrMat)
    # Assign the parameters
    branch, trunk = a.branch_net, a.trunk_net

    #= Dot product needs a dim to contract
    However, we perform the transformations by the NNs always in the first dim
    so we need to adjust (i.e. transpose) one of the inputs,
    which we do on the branch input here =#
    return branch(x)' * trunk(y)
end

# Sensors stay the same and shouldn't be batched
(a::DeepONet)(x::AbstractArray, y::AbstractArray) =
  throw(ArgumentError("Sensor locations fed to trunk net can't be batched."))

# Print nicely
function Base.show(io::IO, l::DeepONet)
    print(io, "DeepONet with\nbranch net: (",l.branch_net)
    print(io, ")\n")
    print(io, "Trunk net: (", l.trunk_net)
    print(io, ")\n")
end
