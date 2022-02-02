using Plots.PlotMeasures
using Plots

struct Simulator
    # computational size
    max_x::Float64
    max_y::Float64
    max_t::Float64

    # wave length
    λ::Float64

    # radious of defect
    r::Float64

    # discretization
    nx::Int64
    ny::Int64

    # ##### internal field #####

    # speed of light
    c::Float64

    # difference
    Δx::Float64
    Δy::Float64

    # discretization of time is determined by Δx and Δy
    Δt::Float64
    nt::Int64

    # permittivity
    ϵ::Matrix{Float64}
    ϵx::Matrix{Float64}
    ϵy::Matrix{Float64}

    # permeability
    μ::Float64
    μx::Float64
    μy::Float64

    # wavenumber
    k::Float64
end

function Simulator(;
    max_x=3e-6, max_y=10e-6, max_t=1e-12,
    λ=2.04e-6,
    r=0.45e-6,
    nx=120, ny=400
)
    c = 299792458

    Δx = max_x/nx
    Δy = max_x/ny
    Δt = 1 / c / √(1/Δx^2 + 1/Δy^2)
    nt = round(Int, max_t/Δt)

    μ = 1
    μx = c * Δt/Δx / μ
    μy = c * Δt/Δy / μ

    k = 2π/λ

    s = Simulator(
        max_x, max_y, max_t,
        λ,
        r,
        nx, ny,

        c, Δx, Δy, Δt, nt,
        ones(nx, ny), ones(nx, ny), ones(nx, ny),
        μ, μx, μy,

        k
    )
    set_rand_ϵ!(s)

    return s
end

function set_rand_ϵ!(s::Simulator)
    ϵ = 9 * ones(s.nx, s.ny)

    n = rand(1:5)
    xs = s.max_x .* rand(n)
    ys = s.max_y .* rand(n)
    rs = s.r .* rand(n)

    function in_circle(i, j)
        return true in [√((i*s.Δx - xs[c])^2 + (j*s.Δy - ys[c])^2) < rs[c] for c in 1:n]
    end

    for i in 1:s.nx
        for j in 1:s.ny
            in_circle(i, j) && (ϵ[i, j] = 1)
        end
    end

    s.ϵ .= ϵ
    s.ϵx .= s.c * s.Δt/s.Δx ./ ϵ
    s.ϵy .= s.c * s.Δt/s.Δy ./ ϵ

    return s
end

function plot_ϵ(s::Simulator)
    plotly()

    return heatmap(
		LinRange(0, s.max_x, s.nx), LinRange(0, s.max_y, s.ny), s.ϵ',
		color=:coolwarm,
		size=(350, 750), left_margin=-100px
	)
end

function plot_sim(s::Simulator)
    plotly()

    contour!(
        LinRange(0, s.max_x, s.nx), LinRange(0, s.max_y, s.ny), s.ϵ',
        color=:coolwarm,
    )
end
