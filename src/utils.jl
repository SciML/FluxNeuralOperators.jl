@inline function __project(b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return dropdims(sum(b_ .* t; dims=1); dims=1) # N x nb
end

@inline function __project(b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2}
    # b : p x u x nb
    # t : p x N x nb
    if size(b, 2) == 1 || size(t, 2) == 1
        return sum(b .* t; dims=1) # 1 x N x nb
    else
        return batched_matmul(batched_adjoint(b), t) # u x N x b
    end
end

@inline function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3},
        additional::Nothing) where {T1, T2, N}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), u_size..., 1, size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t, 1), ones(eltype(u_size), length(u_size))..., size(t)[2:end]...)
    # p x (1,1,1...) x N x nb

    return dropdims(sum(b_ .* t_; dims=1); dims=1) # u_size x N x nb
end

@inline function __project(
        b::AbstractArray{T1, 2}, t::AbstractArray{T2, 3}, additional::T) where {T1, T2, T}
    # b : p x nb
    # t : p x N x nb
    b_ = reshape(b, size(b, 1), 1, size(b, 2)) # p x 1 x nb
    return additional(b_ .* t) # p x N x nb => out_dims x N x nb
end

@inline function __project(
        b::AbstractArray{T1, 3}, t::AbstractArray{T2, 3}, additional::T) where {T1, T2, T}
    # b : p x u x nb
    # t : p x N x nb

    if size(b, 2) == 1 || size(t, 2) == 1
        return additional(b .* t) # p x N x nb => out_dims x N x nb
    else
        b_ = reshape(b, size(b)[1:2]..., 1, size(b, 3)) # p x u x 1 x nb
        t_ = reshape(t, size(t, 1), 1, size(t)[2:end]...) # p x 1 x N x nb

        return additional(b_ .* t_) # p x u x N x nb => out_size x N x nb
    end
end

@inline function __project(b::AbstractArray{T1, N}, t::AbstractArray{T2, 3},
        additional::T) where {T1, T2, N, T}
    # b : p x u_size x nb
    # t : p x N x nb
    u_size = size(b)[2:(end - 1)]

    b_ = reshape(b, size(b, 1), u_size..., 1, size(b)[end])
    # p x u_size x 1 x nb

    t_ = reshape(t, size(t, 1), ones(eltype(u_size), length(u_size))..., size(t)[2:end]...)
    # p x (1,1,1...) x N x nb

    return additional(b_ .* t_) # p x u_size x N x nb => out_size x N x nb
end
