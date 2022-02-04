module FieldDistributionNonuniformMedium

using Plots.PlotMeasures
using Plots

const C = 299792458

struct Bound
    max_x::Float64
    max_y::Float64
    max_t::Float64
end

struct Discretizer
    nx::Int64
    ny::Int64

    Δx::Float64
    Δy::Float64

    Δt::Float64
    nt::Int64
end

function Discretizer(nx, ny, b::Bound)
    Δx = b.max_x / nx
    Δy = b.max_y / ny

    Δt = 1 / C / √(1/Δx^2 + 1/Δy^2)
    nt = round(Int, b.max_t/Δt)

    return Discretizer(nx, ny, Δx, Δy, Δt, nt)
end

struct Light
    λ::Float64
    k::Float64
end

function Light(λ)
    return Light(λ, 2π/λ)
end

struct Permittivity
    ϵ::Matrix{Float64}
    ϵx::Matrix{Float64}
    ϵy::Matrix{Float64}
end

function RandPermittivity(n::Integer, r::Float64, b::Bound, d::Discretizer)
    ϵ = 9 * ones(d.nx, d.ny)

    xs = b.max_x .* rand(n)
    ys = b.max_y .* rand(n)
    rs = r .* rand(n)

    in_circle(i, j) = true in [
        √((i*d.Δx - xs[c])^2 + (j*d.Δy - ys[c])^2) < rs[c] for c in 1:n
    ]

    for i in 1:d.nx, j in 1:d.ny
        in_circle(i, j) && (ϵ[i, j] = 1)
    end

    ϵx = C * d.Δt/d.Δx ./ ϵ
    ϵy = C * d.Δt/d.Δy ./ ϵ

    return Permittivity(ϵ, ϵx, ϵy)
end

struct Permeability
    μ::Float64
    μx::Float64
    μy::Float64
end

function Permeability(μ, d::Discretizer)
    μx = C * d.Δt/d.Δx / μ
    μy = C * d.Δt/d.Δy / μ

    return Permeability(μ, μx, μy)
end

mutable struct Simulator
    bound::Bound
    discretizer::Discretizer
    light::Light
    permittivity::Permittivity
    permeability::Permeability

    ez::Matrix{Float64}
    hx::Matrix{Float64}
    hy::Matrix{Float64}

    t::Int64
end

function Simulator(;
    max_x=3e-6, max_y=10e-6, max_t=5e-12,
    nx=300, ny=1000,
    λ=2.04e-6,
    n=rand(1:5), r=0.45e-6,
    μ=1
)
    bound = Bound(max_x, max_y, max_t)
    discretizer = Discretizer(nx, ny, bound)
    light = Light(λ)
    permittivity = RandPermittivity(n, r, bound, discretizer)
    permeability = Permeability(μ, discretizer)

    Δx, Δt = discretizer.Δx, discretizer.Δt

    ez = zeros(Float64, nx, ny)
    ez[2:nx, 1] .= 0.1exp.(
        -(Δx * ((2:nx) .- nx/2)).^2 ./
        (max_x/4)^2
    ) * sin(light.k * C*Δt)

    return Simulator(
        bound,
        discretizer,
        light,
        permittivity,
        permeability,

        ez,
        zeros(Float64, nx, ny),
        zeros(Float64, nx, ny),

        0
    )
end

function next!(s::Simulator)
    max_x = s.bound.max_x
    nx, ny, Δx, Δt = s.discretizer.nx, s.discretizer.ny, s.discretizer.Δx, s.discretizer.Δt
    k = s.light.k
    ϵx, ϵy = s.permittivity.ϵx, s.permittivity.ϵy
    μx, μy = s.permeability.μx, s.permeability.μy

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

    return s
end

function simulate!(s::Simulator)
    for _ in 1:(s.discretizer.nt)
        next!(s)
    end

    return s
end

function plot_ϵ(s::Simulator; size=(350, 750), left_margin=-100px)
    plotly()

    max_x, max_y = s.bound.max_x, s.bound.max_y
    nx, ny = s.discretizer.nx, s.discretizer.ny
    ϵ = s.permittivity.ϵ

    return heatmap(
		LinRange(0, max_x, nx), LinRange(0, max_y, ny), ϵ',
		color=:algae,
		size=size, left_margin=left_margin
	)
end

function plot_e_field(s::Simulator; size=(300, 750), left_margin=-100px)
    plotly()

    max_x, max_y = s.bound.max_x, s.bound.max_y
    nx, ny = s.discretizer.nx, s.discretizer.ny
    ez = s.ez
    ϵ = s.permittivity.ϵ

    lim = maximum(abs.(ez))
    p = heatmap(
		LinRange(0, max_x, nx), LinRange(0, max_y, ny), ez',
		color=:coolwarm, clim=(-lim, lim), colorbar=false,
		size=size, left_margin=left_margin
	)
    lim_ϵ = maximum(abs.(ϵ))
    p = contour!(
        p,
        LinRange(0, max_x, nx), LinRange(0, max_y, ny), lim .* ϵ' ./ lim_ϵ,
        color=:algae, colorbar=false
    )

    return p
end

function get_grid(s::Simulator)
    Δx, Δy, nx, ny = s.discretizer.Δx, s.discretizer.Δy, s.discretizer.nx, s.discretizer.ny

    return cat(
        repeat(Δx * ((1:nx) .- nx/2), 1, ny, 1),
        repeat(Δy * (transpose(1:ny) .- Δy/2), nx, 1, 1),
        dims=3
    )
end

function gen_data(; nx=60, ny=200, n=10000)

    for i in 1:n
        @info "data $i:"
        @time begin
            s = Simulator(nx=nx, ny=ny)
            simulate!(s)
        end
    end
end

end # module
