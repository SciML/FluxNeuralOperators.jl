function apply_pattern(
        x_tr::AbstractArray{T1, N}, weights::AbstractArray{T2, 3}) where {T1, T2, N}
    x_size = size(x_tr)
    x_flat = reshape(x_tr, :, x_size[N - 1], x_size[N])

    x_flat_t = permutedims(x_flat, (2, 3, 1))                               # i x b x m
    x_weighted = permutedims(batched_matmul(weights, x_flat_t), (3, 1, 2))  # m x o x b

    return reshape(x_weighted, x_size[1:(N - 2)]..., size(x_weighted)[2:3]...)
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

function expand_pad_dims(pad_dims::Dims{N}) where {N}
    return ntuple(i -> isodd(i) ? 0 : pad_dims[i ÷ 2], 2N)
end

@non_differentiable expand_pad_dims(::Any)
