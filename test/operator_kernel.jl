@testset "1D OperatorConv" begin
    modes = (16,)
    ch = 64 => 128

    m = Chain(Dense(2, 64),
              OperatorConv(ch, modes, FourierTransform))
    @test ndims(OperatorConv(ch, modes, FourierTransform)) == 1
    @test repr(OperatorConv(ch, modes, FourierTransform)) ==
          "OperatorConv(64 => 128, (16,), FourierTransform, permuted=false)"

    𝐱 = rand(Float32, 2, 1024, 5)
    @test size(m(𝐱)) == (128, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "permuted 1D OperatorConv" begin
    modes = (16,)
    ch = 64 => 128

    m = Chain(Conv((1,), 2 => 64),
              OperatorConv(ch, modes, FourierTransform, permuted = true))
    @test ndims(OperatorConv(ch, modes, FourierTransform, permuted = true)) == 1
    @test repr(OperatorConv(ch, modes, FourierTransform, permuted = true)) ==
          "OperatorConv(64 => 128, (16,), FourierTransform, permuted=true)"

    𝐱 = rand(Float32, 2, 1024, 5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 128, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 128, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "1D OperatorKernel" begin
    modes = (16,)
    ch = 64 => 128

    m = Chain(Dense(2, 64),
              OperatorKernel(ch, modes, FourierTransform))
    @test repr(OperatorKernel(ch, modes, FourierTransform)) ==
          "OperatorKernel(64 => 128, (16,), FourierTransform, σ=identity, permuted=false)"

    𝐱 = rand(Float32, 2, 1024, 5)
    @test size(m(𝐱)) == (128, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "permuted 1D OperatorKernel" begin
    modes = (16,)
    ch = 64 => 128

    m = Chain(Conv((1,), 2 => 64),
              OperatorKernel(ch, modes, FourierTransform, permuted = true))
    @test repr(OperatorKernel(ch, modes, FourierTransform, permuted = true)) ==
          "OperatorKernel(64 => 128, (16,), FourierTransform, σ=identity, permuted=true)"

    𝐱 = rand(Float32, 2, 1024, 5)
    𝐱 = permutedims(𝐱, (2, 1, 3))
    @test size(m(𝐱)) == (1024, 128, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 1024, 128, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "2D OperatorConv" begin
    modes = (10, 10)
    ch = 64 => 64

    m = Chain(Dense(1, 64),
              OperatorConv(ch, modes, FourierTransform))
    @test ndims(OperatorConv(ch, modes, FourierTransform)) == 2

    𝐱 = rand(Float32, 1, 22, 22, 5)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "permuted 2D OperatorConv" begin
    modes = (10, 10)
    ch = 64 => 64

    m = Chain(Conv((1, 1), 1 => 64),
              OperatorConv(ch, modes, FourierTransform, permuted = true))
    @test ndims(OperatorConv(ch, modes, FourierTransform, permuted = true)) == 2

    𝐱 = rand(Float32, 1, 22, 22, 5)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "2D OperatorKernel" begin
    modes = (10, 10)
    ch = 64 => 64

    m = Chain(Dense(1, 64),
              OperatorKernel(ch, modes, FourierTransform))

    𝐱 = rand(Float32, 1, 22, 22, 5)
    @test size(m(𝐱)) == (64, 22, 22, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 64, 22, 22, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "permuted 2D OperatorKernel" begin
    modes = (10, 10)
    ch = 64 => 64

    m = Chain(Conv((1, 1), 1 => 64),
              OperatorKernel(ch, modes, FourierTransform, permuted = true))

    𝐱 = rand(Float32, 1, 22, 22, 5)
    𝐱 = permutedims(𝐱, (2, 3, 1, 4))
    @test size(m(𝐱)) == (22, 22, 64, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 22, 22, 64, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "SpectralConv" begin
    modes = (16, 16)
    ch = 64 => 64
    permuted = false

    @test SpectralConv(ch, modes) isa OperatorConv
    @test SpectralConv(ch, modes).transform isa FourierTransform
end

@testset "1D OperatorConv with ChebyshevTransform" begin
    modes = (16,)
    ch = 64 => 128

    m = Chain(Dense(2, 64),
              OperatorConv(ch, modes, ChebyshevTransform))
    @test ndims(OperatorConv(ch, modes, ChebyshevTransform)) == 1
    @test repr(OperatorConv(ch, modes, ChebyshevTransform)) ==
          "OperatorConv(64 => 128, (16,), ChebyshevTransform, permuted=false)"

    𝐱 = rand(Float32, 2, 1024, 5)
    @test size(m(𝐱)) == (128, 1024, 5)

    loss(x, y) = Flux.mse(m(x), y)
    data = [(𝐱, rand(Float32, 128, 1024, 5))]
    Flux.train!(loss, Flux.params(m), data, Flux.Adam())
end

@testset "SparseKernel" begin
    T = Float32
    k = 3
    batch_size = 32

    @testset "1D SparseKernel" begin
        α = 4
        c = 1
        in_chs = 20
        X = rand(T, in_chs, c * k, batch_size)

        l1 = SparseKernel1D(k, α, c)
        Y = l1(X)
        @test l1 isa SparseKernel{1}
        @test size(Y) == size(X)

        gs = gradient(() -> sum(l1(X)), Flux.params(l1))
        @test length(gs.grads) == 4
    end

    @testset "2D SparseKernel" begin
        α = 4
        c = 3
        Nx = 5
        Ny = 7
        X = rand(T, Nx, Ny, c * k^2, batch_size)

        l2 = SparseKernel2D(k, α, c)
        Y = l2(X)
        @test l2 isa SparseKernel{2}
        @test size(Y) == size(X)

        gs = gradient(() -> sum(l2(X)), Flux.params(l2))
        @test length(gs.grads) == 4
    end

    @testset "3D SparseKernel" begin
        α = 4
        c = 3
        Nx = 5
        Ny = 7
        Nz = 13
        X = rand(T, Nx, Ny, Nz, α * k^2, batch_size)

        l3 = SparseKernel3D(k, α, c)
        Y = l3(X)
        @test l3 isa SparseKernel{3}
        @test size(Y) == (Nx, Ny, Nz, c * k^2, batch_size)

        gs = gradient(() -> sum(l3(X)), Flux.params(l3))
        @test length(gs.grads) == 4
    end
end
