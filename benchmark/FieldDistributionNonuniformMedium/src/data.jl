using ProgressMeter
using JLD2
using Flux

function gen_data(; nx=60, ny=200, n=7000)
    xs = Array{Float64, 4}(undef, nx-2, ny-2, 3, n)
    ys = Array{Float64, 4}(undef, nx-2, ny-2, 1, n)
    p = Progress(n)
    Threads.@threads for i in 1:n
        s = Simulator(nx=nx, ny=ny)
        simulate!(s)

        xs[:, :, 1, i] .= s.permittivity.ϵ[2:(nx-1), 2:(ny-1)]
        xs[:, :, 2:3, i] .= build(s.grid)[2:(nx-1), 2:(ny-1), :]
        ys[:, :, 1, i] .= s.ez[2:(nx-1), 2:(ny-1)]

        next!(p)
    end
    jldsave(joinpath(mkpath(joinpath(@__DIR__, "..", "data")), "data.jld2"); xs, ys)
end

function get_dataloader(; ratio=0.9, batchsize=10)
    f = jldopen(joinpath(@__DIR__, "..", "data", "data.jld2"))
    xs, ys = permutedims(f["xs"], (3, 1, 2, 4)), permutedims(f["ys"],(3, 1, 2, 4))
    close(f)

    n = round(Int, ratio*size(ys, 4))
    loader_train = Flux.DataLoader((xs[:, :, :, 1:n], ys[:, :, :, 1:n]), batchsize=batchsize, shuffle=true)
    loader_test = Flux.DataLoader((xs[:, :, :, (n+1):end], ys[:, :, :, (n+1):end]), batchsize=batchsize, shuffle=true)

    return loader_train, loader_test
end
