function __project(
        b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}, ::NoOpLayer, _) where {T1, T2}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return dropdims(sum(b_ .* t; dims=1); dims=1), () # N x nb
end

function __project(
        b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}, ::NoOpLayer, _) where {T1, T2}
    # b : p x u x nb
    # t : p x N x nb
    if size(b, 2) == 1 || size(t, 2) == 1
        return sum(b .* t; dims=1), () # 1 x N x nb
    else
        return batched_matmul(batched_adjoint(b), t), () # u x N x b
    end
end

function __project(
        b::AbstractArray{T1, N}, t::AbstractArray{T2, 3}, ::NoOpLayer, _) where {T1, T2, N}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), u_size..., 1, size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t, 1), ones(eltype(u_size), length(u_size))..., size(t)[2:end]...)
    # p x (1,1,1...) x N x nb

    return dropdims(sum(b_ .* t_; dims=1); dims=1), () # u_size x N x nb
end

function __project(b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3},
        additional::T, params) where {T1, T2, T}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return additional(b_ .* t, params.ps, params.st) # p x N x nb => out_dims x N x nb
end

function __project(b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3},
        additional::T, params) where {T1, T2, T}
    # b : p x u x nb
    # t : p x N x nb

    if size(b, 2) == 1 || size(t, 2) == 1
        return additional(b .* t, params.ps, params.st) # p x N x nb => out_dims x N x nb
    else
        b_ = reshape(b, size(b)[1:2]..., 1, size(b, 3)) # p x u x 1 x nb
        t_ = reshape(t, size(t, 1), 1, size(t)[2:end]...) # p x 1 x N x nb

        return additional(b_ .* t_, params.ps, params.st) # p x u x N x nb => out_size x N x nb
    end
end

function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3},
        additional::T, params) where {T1, T2, N, T}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), u_size..., 1, size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t, 1), ones(eltype(u_size), length(u_size))..., size(t)[2:end]...)
    # p x (1,1,1...) x N x nb

    return additional(b_ .* t_, params.ps, params.st) # p x u_size x N x nb => out_size x N x nb
end

function __batch_vectorize(x::AbstractArray{T, N}) where {T, N}
    dim_length = ndims(x) - 1
    nb = size(x)[end]

    slice = [Colon() for _ in 1:dim_length]
    return reduce(hcat, [vec(view(x, slice..., i)) for i in 1:nb])
end

function __merge(x::AbstractArray{T1, 2}, y::AbstractArray{T2, 2}) where {T1, T2}
    return cat(x, y; dims=1)
end

function __merge(x::AbstractArray{T1, N1}, y::AbstractArray{T2, 2}) where {T1, T2, N1}
    x_ = __batch_vectorize(x)
    return vcat(x_, y)
end

function __merge(x::AbstractArray{T1, 2}, y::AbstractArray{T2, N2}) where {T1, T2, N2}
    y_ = __batch_vectorize(y)
    return vcat(x, y_)
end

function __merge(x::AbstractArray{T1, N1}, y::AbstractArray{T2, N2}) where {T1, T2, N1, N2}
    x_ = __batch_vectorize(x)
    y_ = __batch_vectorize(y)
    return vcat(x_, y_)
end

function add_act(act::F, x1, x2) where {F}
    y = x1 .+ x2
    act = NNlib.fast_act(act, y)
    return fast_activation!!(act, y)
end

@concrete struct Fix1 <: Function
    f
    x
end

(f::Fix1)(args...) = f.f(f.x, args...)
