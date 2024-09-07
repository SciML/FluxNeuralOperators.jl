function operator_conv(x, tform::AbstractTransform, weights)
    x_t = transform(tform, x)
    x_tr = truncate_modes(tform, x_t)
    x_p = __apply_pattern(x_tr, weights)
    x_padded = __pad_modes(x_p, size(x_t)[1:(end - 2)]..., size(x_p)[(end - 1):end]...)
    return inverse(tform, x_padded, size(x))
end

function __apply_pattern(
        x_tr::AbstractArray{T1, N}, weights::AbstractArray{T2, 3}) where {T1, T2, N}
    x_size = size(x_tr)
    x_flat = reshape(x_tr, :, x_size[N - 1], x_size[N])

    x_flat_t = permutedims(x_flat, (2, 3, 1))                               # i x b x m
    x_weighted = permutedims(batched_matmul(weights, x_flat_t), (3, 1, 2))  # m x o x b

    return reshape(x_weighted, x_size[1:(N - 2)]..., size(x_weighted)[2:3]...)
end

__pad_modes(x, dims::Integer...) = __pad_modes(x, dims)
__pad_modes(x, dims::NTuple) = __pad_modes!(similar(x, dims), x)

function __pad_modes!(x_padded::AbstractArray, x::AbstractArray)
    fill!(x_padded, eltype(x)(0))
    x_padded[map(d -> 1:d, size(x))...] .= x
    return x_padded
end

function CRC.rrule(::typeof(__pad_modes), x::AbstractArray, dims::NTuple)
    ∇pad_modes = let x = x
        ∂y -> (NoTangent(), view(∂y, map(Base.OneTo, size(x))...), NoTangent())
    end
    return __pad_modes(x, dims), ∇pad_modes
end
