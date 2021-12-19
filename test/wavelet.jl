using NeuralOperators

T = Float32
k = 10
c = 1
in_chs = 20
batch_size = 32


l = NeuralOperators.SparseKernel1d(k, c)

X = rand(T, c*k, in_chs, batch_size)
Y = l(X)
