using Plots.PlotMeasures
using Plots

function plot_ϵ(s::Simulator, grid::Grid; figsize=(350, 750), left_margin=-100px)
    plotly()

    max_x, max_y = boundary(grid)
    nx, ny = size(grid)
    ϵ = s.permittivity.ϵ

    return heatmap(
		LinRange(0, max_x, nx), LinRange(0, max_y, ny), ϵ',
		color=:algae,
		size=figsize, left_margin=left_margin
	)
end

function plot_e_field(s::Simulator, grid::Grid; figsize=(300, 750), left_margin=-100px)
    plotly()

    max_x, max_y = boundary(grid)
    nx, ny = size(grid)
    ez = s.ez
    ϵ = s.permittivity.ϵ

    lim = maximum(abs.(ez))
    p = heatmap(
		LinRange(0, max_x, nx), LinRange(0, max_y, ny), ez',
		color=:coolwarm, clim=(-lim, lim), colorbar=false,
		size=figsize, left_margin=left_margin
	)
    lim_ϵ = maximum(abs.(ϵ))
    p = contour!(
        p,
        LinRange(0, max_x, nx), LinRange(0, max_y, ny), lim .* ϵ' ./ lim_ϵ,
        color=:algae, colorbar=false
    )

    return p
end
