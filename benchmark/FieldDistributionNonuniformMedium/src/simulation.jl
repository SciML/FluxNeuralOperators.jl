export 
    Grid,
    boundary,
    build,
    Simulator,
    simulate!


const C = 299792458

struct Grid{T<:Real}
    nx::Int
    ny::Int
    nt::Int

    Δx::T
    Δy::T
    Δt::T

    max_x::T
    max_y::T
    max_t::T
end

function Grid(nx, ny, max_x, max_y, max_t)
    Δx = max_x / nx
    Δy = max_y / ny

    Δt = 1 / C / √(1/Δx^2 + 1/Δy^2)
    nt = round(Int, max_t/Δt)

    return Grid(nx, ny, nt, Δx, Δy, Δt, max_x, max_y, max_t)
end

function build(grid::Grid)
    Δx, Δy, nx, ny = grid.Δx, grid.Δy, grid.nx, grid.ny

    return cat(
        repeat(Δx * ((1:nx) .- nx/2), 1, ny, 1),
        repeat(Δy * (transpose(1:ny) .- Δy/2), nx, 1, 1),
        dims=3
    )
end

Base.size(grid::Grid) = (grid.nx, grid.ny)
Base.size(grid::Grid, d) = d::Integer <= 2 ? size(grid)[d] : 1
Base.axes(grid::Grid) = (Base.OneTo(grid.nx), Base.OneTo(grid.ny))
Base.axes(grid::Grid, d) = d::Integer <= 2 ? axes(grid)[d] : 1
boundary(grid::Grid) = (grid.max_x, grid.max_y)
boundary(grid::Grid, d) = d::Integer <= 2 ? boundary(grid)[d] : 1

struct Light{T<:Real}
    λ::T
    k::T
end

function Light(λ)
    return Light(λ, 2π/λ)
end

struct Permittivity{T<:AbstractMatrix}
    ϵ::T
    ϵx::T
    ϵy::T
end

function implant(::Type{Permittivity}, xs::AbstractVector, ys::AbstractVector, rs::AbstractVector, grid::Grid)
    ϵ = 9 * ones(size(grid))

    length(xs) == length(ys) == length(rs) || throw(DimensionMismatch("xs, ys, rs must have same length"))

    n = length(xs)

    in_circle(i, j) = true in [
        √((i*grid.Δx - xs[c])^2 + (j*grid.Δy - ys[c])^2) < rs[c] for c in 1:n
    ]

    for i in 1:grid.nx, j in 1:grid.ny
        in_circle(i, j) && (ϵ[i, j] = 1)
    end

    ϵx = C * grid.Δt/grid.Δx ./ ϵ
    ϵy = C * grid.Δt/grid.Δy ./ ϵ

    return Permittivity(ϵ, ϵx, ϵy)
end

function Base.rand(Tϵ::Type{Permittivity}, n::Integer, r::Real, grid::Grid)
    xs = grid.max_x .* rand(n)
    ys = grid.max_y .* rand(n)
    rs = r .* rand(n)

    return implant(Tϵ, xs, ys, rs, grid)
end
struct Permeability{T<:Real}
    μ::T
    μx::T
    μy::T
end

function Permeability(μ, grid::Grid)
    μx = C * grid.Δt/grid.Δx / μ
    μy = C * grid.Δt/grid.Δy / μ

    return Permeability(μ, μx, μy)
end

mutable struct Simulator{T<:AbstractArray}
    grid::Grid
    light::Light
    permittivity::Permittivity
    permeability::Permeability

    ez::T
    hx::T
    hy::T

    t::Int
end

function Simulator(;
    max_x=3e-6, max_y=10e-6, max_t=5e-12,
    nx=300, ny=1000,
    λ=2.04e-6,
    n=rand(1:5), r=0.45e-6,
    μ=1.
)
    grid = Grid(nx, ny, max_x, max_y, max_t)
    light = Light(λ)
    permittivity = rand(Permittivity, n, r, grid)
    permeability = Permeability(μ, grid)

    Δx, Δt = grid.Δx, grid.Δt

    ez = zeros(Float64, size(grid))
    ez[2:nx, 1] .= 0.1exp.(
        -(Δx * ((2:nx) .- nx/2)).^2 ./
        (max_x/4)^2
    ) * sin(light.k * C*Δt)

    return Simulator(
        grid,
        light,
        permittivity,
        permeability,

        ez,
        zeros(Float64, size(grid)),
        zeros(Float64, size(grid)),

        0
    )
end

function simulate!(s::Simulator)
    nx, ny = size(s.grid)
    Δx, Δt, max_x = s.grid.Δx, s.grid.Δt, s.grid.max_x
    k = s.light.k
    ϵx, ϵy = s.permittivity.ϵx, s.permittivity.ϵy
    μx, μy = s.permeability.μx, s.permeability.μy

    for _ in 1:(s.grid.nt)
        s.ez[2:nx, 1] .+= 0.1exp.(
            -(Δx * ((2:nx) .- nx/2)).^2 ./
            (max_x/4)^2
        ) * sin(k * C*Δt*s.t)

        s.hx[2:(nx-1), 2:(ny-1)] .+= -μx*(s.ez[2:(nx-1), 2:(ny-1)] - s.ez[2:(nx-1), 1:(ny-2)])
        s.hy[2:(nx-1), 2:(ny-1)] .+= +μy*(s.ez[2:(nx-1), 2:(ny-1)] - s.ez[1:(nx-2), 2:(ny-1)])

        s.ez[2:(nx-1), 2:(ny-1)] .+=
            ϵx[2:(nx-1), 2:(ny-1)].*(s.hy[3:nx, 2:(ny-1)] - s.hy[2:(nx-1), 2:(ny-1)]) -
            ϵy[2:(nx-1), 2:(ny-1)].*(s.hx[2:(nx-1), 3:ny] - s.hx[2:(nx-1), 2:(ny-1)])

        s.t += 1
    end

    return s
end
