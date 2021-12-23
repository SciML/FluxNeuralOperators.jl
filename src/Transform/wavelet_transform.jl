export
    SparseKernel,
    SparseKernel1D,
    SparseKernel2D,
    SparseKernel3D


struct SparseKernel{N,T,S}
    conv_blk::T
    out_weight::S
end

function SparseKernel(filter::NTuple{N,T}, ch::Pair{S, S}; init=Flux.glorot_uniform) where {N,T,S}
    input_dim, emb_dim = ch
    conv = Conv(filter, input_dim=>emb_dim, relu; stride=1, pad=1, init=init)
    W_out = Dense(emb_dim, input_dim; init=init)
    return SparseKernel{N,typeof(conv),typeof(W_out)}(conv, W_out)
end

function SparseKernel1D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k
    emb_dim = 128
    return SparseKernel((3, ), input_dim=>emb_dim; init=init)
end

function SparseKernel2D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k^2
    emb_dim = α*k^2
    return SparseKernel((3, 3), input_dim=>emb_dim; init=init)
end

function SparseKernel3D(k::Int, α, c::Int=1; init=Flux.glorot_uniform)
    input_dim = c*k^2
    emb_dim = α*k^2
    conv = Conv((3, 3, 3), emb_dim=>emb_dim, relu; stride=1, pad=1, init=init)
    W_out = Dense(emb_dim, input_dim; init=init)
    return SparseKernel{3,typeof(conv),typeof(W_out)}(conv, W_out)
end

Flux.@functor SparseKernel

function (l::SparseKernel)(X::AbstractArray)
    bch_sz, _, dims_r... = reverse(size(X))
    dims = reverse(dims_r)

    X_ = l.conv_blk(X)  # (dims..., emb_dims, B)
    X_ = reshape(X_, prod(dims), :, bch_sz)  # (prod(dims), emb_dims, B)
    Y = l.out_weight(batched_transpose(X_))  # (in_dims, prod(dims), B)
    Y = reshape(batched_transpose(Y), dims..., :, bch_sz)  # (dims..., in_dims, B)
    return collect(Y)
end


struct MWT_CZ1d{T,S,R,Q,P}
    k::Int
    L::Int
    A::T
    B::S
    C::R
    T0::Q
    ec_s::P
    ec_d::P
    rc_e::P
    rc_o::P
end

function MWT_CZ1d(k::Int=3, α::Int=5, L::Int=0, c::Int=1; base::Symbol=:legendre, init=Flux.glorot_uniform)
    H0, H1, G0, G1, ϕ0, ϕ1 = get_filter(base, k)
    H0r = zero_out!(H0 * ϕ0)
    G0r = zero_out!(G0 * ϕ0)
    H1r = zero_out!(H1 * ϕ1)
    G1r = zero_out!(G1 * ϕ1)

    dim = c*k
    A = SpectralConv(dim=>dim, (α,); init=init)
    B = SpectralConv(dim=>dim, (α,); init=init)
    C = SpectralConv(dim=>dim, (α,); init=init)
    T0 = Dense(k, k)

    ec_s = vcat(H0', H1')
    ec_d = vcat(G0', G1')
    rc_e = vcat(H0r, G0r)
    rc_o = vcat(H1r, G1r)
    return MWT_CZ1d(k, L, A, B, C, T0, ec_s, ec_d, rc_e, rc_o)
end

function wavelet_transform(l::MWT_CZ1d, X::AbstractArray{T,4}) where {T}
    N = size(X, 3)
    Xa = vcat(view(X, :, :, 1:2:N, :), view(X, :, :, 2:2:N, :))
    d = NNlib.batched_mul(Xa, l.ec_d)
    s = NNlib.batched_mul(Xa, l.ec_s)
    return d, s
end

function even_odd(l::MWT_CZ1d, X::AbstractArray{T,4}) where {T}
    bch_sz, N, dims_r... = reverse(size(X))
    dims = reverse(dims_r)
    @assert dims[1] == 2*l.k
    Xₑ = NNlib.batched_mul(X, l.rc_e)
    Xₒ = NNlib.batched_mul(X, l.rc_o)
#         x = torch.zeros(B, N*2, c, self.k,
#             device = x.device)
#         x[..., ::2, :, :] = x_e
#         x[..., 1::2, :, :] = x_o
    return X
end

function (l::MWT_CZ1d)(X::T) where {T<:AbstractArray}
    bch_sz, N, dims_r... = reverse(size(X))
    ns = floor(log2(N))
    stop = ns - l.L

    # decompose
    Ud = T[]
    Us = T[]
    for i in 1:stop
        d, X = wavelet_transform(l, X)
        push!(Ud, l.A(d)+l.B(d))
        push!(Us, l.C(d))
    end
    X = l.T0(X)

    # reconstruct
    for i in stop:-1:1
        X += Us[i]
        X = vcat(X, Ud[i])  # x = torch.cat((x, Ud[i]), -1)
        X = even_odd(l, X)
    end
    return X
end
