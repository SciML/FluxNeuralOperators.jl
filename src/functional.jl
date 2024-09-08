function operator_conv(x, tform::AbstractTransform, weights)
    x_t = transform(tform, x)
    x_tr = truncate_modes(tform, x_t)
    x_p = __apply_pattern(x_tr, weights)

    pad_dims = size(x_t)[1:(end - 2)] .- size(x_p)[1:(end - 2)]
    x_padded = NNlib.pad_constant(x_p, expand_pad_dims(pad_dims), false;
        dims=ntuple(identity, ndims(x_p) - 2))::typeof(x_p)

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
