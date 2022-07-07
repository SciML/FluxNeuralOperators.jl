export GraphKernel

"""
    GraphKernel(κ, ch, σ=identity)

Graph kernel layer.

## Arguments

* `κ`: A neural network layer for approximation, e.g. a `Dense` layer or a MLP.
* `ch`: Channel size for linear transform, e.g. `32`.
* `σ`: Activation function.

## Keyword Arguments

* `init`: Initial function to initialize parameters.
"""
struct GraphKernel{A, B, F} <: MessagePassing
    linear::A
    κ::B
    σ::F
end

function GraphKernel(κ, ch::Int, σ = identity; init = Flux.glorot_uniform)
    W = init(ch, ch)
    return GraphKernel(W, κ, σ)
end

Flux.@functor GraphKernel

function GeometricFlux.message(l::GraphKernel, x_i::AbstractArray, x_j::AbstractArray, e_ij)
    return l.κ(vcat(x_i, x_j))
end

function GeometricFlux.update(l::GraphKernel, m::AbstractArray, x::AbstractArray)
    return l.σ.(GeometricFlux._matmul(l.linear, x) + m)
end

function (l::GraphKernel)(el::NamedTuple, X::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    _, V, _ = GeometricFlux.propagate(l, el, nothing, X, nothing, mean, nothing, nothing)
    return V
end

function Base.show(io::IO, l::GraphKernel)
    channel, _ = size(l.linear)
    print(io, "GraphKernel(", l.κ, ", channel=", channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
