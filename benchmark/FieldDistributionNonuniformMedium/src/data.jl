using ProgressMeter
using JLD2

function gen_data(; nx=60, ny=200, n=7000)
    xs = Array{Float64, 4}(undef, nx-2, ny-2, 3, n)
    ys = Array{Float64, 4}(undef, nx-2, ny-2, 1, n)
    p = Progress(n)
    Threads.@threads for i in 1:n
        grid = Grid(; nx=nx, ny=ny)
        s = Simulator(grid)
        simulate!(s, grid)

        xs[:, :, 1, i] .= s.permittivity.ϵ[2:(nx-1), 2:(ny-1)]
        xs[:, :, 2:3, i] .= build(grid)[2:(nx-1), 2:(ny-1), :]
        ys[:, :, 1, i] .= s.ez[2:(nx-1), 2:(ny-1)]

        ProgressMeter.next!(p)
    end
    jldsave(joinpath(mkpath(joinpath(@__DIR__, "..", "data")), "data.jld2"); xs, ys)
end
