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

struct Simulator
    bound::Bound
    discretizer::Discretizer
    light::Light
    permittivity::Permittivity
    permeability::Permeability

    ez::Matrix{Float64}
    hx::Matrix{Float64}
    hy::Matrix{Float64}
end

function Simulator(;
    max_x=3e-6, max_y=10e-6, max_t=1e-12,
    nx=120, ny=400,
    λ=2.04e-6,
    n=rand(1:5), r=0.45e-6,
    μ=1
)
    bound = Bound(max_x, max_y, max_t)
    discretizer = Discretizer(nx, ny, bound)
    light = Light(λ)
    permittivity = RandPermittivity(n, r, bound, discretizer)
    permeability = Permeability(μ, discretizer)

    Δx, Δy, Δt = discretizer.Δx, discretizer.Δy, discretizer.Δt

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
        zeros(Float64, nx, ny)
    )
end

function plot_ϵ(s::Simulator; size=(350, 750), left_margin=-100px)
    plotly()

    max_x, max_y = s.bound.max_x, s.bound.max_y
    nx, ny = s.discretizer.nx, s.discretizer.ny
    ϵ = s.permittivity.ϵ

    return heatmap(
		LinRange(0, max_x, nx), LinRange(0, max_y, ny), ϵ',
		color=:coolwarm,
		size=size, left_margin=left_margin
	)
end

function plot_e_field(s::Simulator; size=(350, 750), left_margin=-100px)
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

    p = contour!(
        p,
        LinRange(0, max_x, nx), LinRange(0, max_y, ny), ϵ',
        color=:grays, colorbar=false
    )

    return p
end
