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

function GeometricFlux.message(l::GraphKernel, x_i, x_j::AbstractArray, e_ij::AbstractArray)
    N = size(x_j, 1)
    K = l.κ(e_ij)
    dims = size(K)[2:end]
    m_ij = GeometricFlux._matmul(reshape(K, N, N, :), reshape(x_j, N, 1, :))

    return reshape(m_ij, N, dims...)
end

function GeometricFlux.update(l::GraphKernel, m::AbstractArray, x::AbstractArray)
    return l.σ.(GeometricFlux._matmul(l.linear, x) + m)
end

function (l::GraphKernel)(el::NamedTuple, X::AbstractArray, E::AbstractArray)
    GraphSignals.check_num_nodes(el.N, X)
    # GraphSignals.check_num_edges(el.E, E)
    _, V, _ = GeometricFlux.propagate(l, el, E, X, nothing, mean, nothing, nothing)

    return V
end

# (wg::WithGraph{<:GraphKernel})(Vt::AbstractArray, E::AbstractArray) = wg(nothing, Vt, E), E

function (wg::WithGraph{<:GraphKernel})(input::AbstractArray)
    el = wg.graph
    Vt, X_with_grid = input[1:end-3, :, :], input[end-2:end, :, :]

    # node features + positional features as edge features
    E = vcat(GeometricFlux._gather(X_with_grid, el.xs),
             GeometricFlux._gather(X_with_grid, el.nbrs))

    return vcat(wg.layer(el, Vt, E), X_with_grid)
end

function Base.show(io::IO, l::GraphKernel)
    channel, _ = size(l.linear)
    print(io, "GraphKernel(", l.κ, ", channel=", channel)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
