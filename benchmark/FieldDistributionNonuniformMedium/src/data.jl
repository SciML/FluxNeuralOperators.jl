using ProgressMeter
using JLD2
using Flux

function gen_data(; t0=0.03e-12, t1=1e-12, max_x=3e-6, max_y=3e-6, nx=60, ny=60, n=10000)
    xs = Array{Float64, 4}(undef, nx-2, ny-2, 3, n)
    ys = Array{Float64, 4}(undef, nx-2, ny-2, 1, n)
    p = Progress(n)
    Threads.@threads for i in 1:n
        s = Simulator(nx=nx, ny=ny, max_x=max_x, max_y=max_x, max_t=t1)

        nt0 = Grid(nx, ny, max_x, max_y, t0).nt
        for t in 1:s.grid.nt
            next!(s)

            if t == nt0
                xs[:, :, 1, i] .= s.ez[2:(nx-1), 2:(ny-1)]
                xs[:, :, 2:3, i] .= build(s.grid)[2:(nx-1), 2:(ny-1), :]
            end
        end
        ys[:, :, 1, i] .= s.ez[2:(nx-1), 2:(ny-1)]

        ProgressMeter.next!(p)
    end

    xs, ys = permutedims(xs, (3, 1, 2, 4)), permutedims(ys, (3, 1, 2, 4))

    jldsave(joinpath(mkpath(joinpath(@__DIR__, "..", "data")), "data.jld2"); xs, ys)
end

function get_dataloader(; ratio=0.9, batchsize=10)
    f = jldopen(joinpath(@__DIR__, "..", "data", "data.jld2"))
    xs, ys = f["xs"], f["ys"]
    close(f)

    n = round(Int, ratio*size(ys, 4))
    loader_train = Flux.DataLoader((xs[:, :, :, 1:n], ys[:, :, :, 1:n]), batchsize=batchsize, shuffle=true)
    loader_test = Flux.DataLoader((xs[:, :, :, (n+1):end], ys[:, :, :, (n+1):end]), batchsize=batchsize, shuffle=true)

    return loader_train, loader_test
end
