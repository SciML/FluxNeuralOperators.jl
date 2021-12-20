using NeuralOperators
using CUDA
using Zygote

CUDA.allowscalar(false)

T = Float32
k = 3
batch_size = 32

α = 4
c = 1
in_chs = 20


l1 = NeuralOperators.SparseKernel1d(k, α, c)
X = rand(T, in_chs, c*k, batch_size)
Y = l1(X)
gradient(x->sum(l1(x)), X)


α = 4
c = 3
Nx = 5
Ny = 7

l2 = NeuralOperators.SparseKernel2d(k, α, c)
X = rand(T, Nx, Ny, c*k^2, batch_size)
Y = l2(X)
gradient(x->sum(l2(x)), X)

Nz = 13

l3 = NeuralOperators.SparseKernel3d(k, α, c)
X = rand(T, Nx, Ny, Nz, α*k^2, batch_size)
Y = l3(X)
gradient(x->sum(l3(x)), X)
