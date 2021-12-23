@testset "SparseKernel" begin
    T = Float32
    k = 3
    batch_size = 32

    @testset "1D SparseKernel" begin
        α = 4
        c = 1
        in_chs = 20
        X = rand(T, in_chs, c*k, batch_size)

        l1 = SparseKernel1D(k, α, c)
        Y = l1(X)
        @test l1 isa SparseKernel{1}
        @test size(Y) == size(X)

        gs = gradient(()->sum(l1(X)), Flux.params(l1))
        @test length(gs.grads) == 4
    end

    @testset "2D SparseKernel" begin
        α = 4
        c = 3
        Nx = 5
        Ny = 7
        X = rand(T, Nx, Ny, c*k^2, batch_size)
    
        l2 = SparseKernel2D(k, α, c)
        Y = l2(X)
        @test l2 isa SparseKernel{2}
        @test size(Y) == size(X)

        gs = gradient(()->sum(l2(X)), Flux.params(l2))
        @test length(gs.grads) == 4
    end

    @testset "3D SparseKernel" begin
        α = 4
        c = 3
        Nx = 5
        Ny = 7
        Nz = 13
        X = rand(T, Nx, Ny, Nz, α*k^2, batch_size)

        l3 = SparseKernel3D(k, α, c)
        Y = l3(X)
        @test l3 isa SparseKernel{3}
        @test size(Y) == (Nx, Ny, Nz, c*k^2, batch_size)

        gs = gradient(()->sum(l3(X)), Flux.params(l3))
        @test length(gs.grads) == 4
    end
end
