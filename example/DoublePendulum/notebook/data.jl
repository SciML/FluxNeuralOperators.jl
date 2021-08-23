### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 194baef2-0417-11ec-05ab-4527ef614024
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 38c9ced5-dcf8-4e03-ac07-7c435687861b
begin
	using DoublePendulum
	using Plots
end

# ╔═╡ 5268feee-bda2-4612-9d4c-a1db424a11c7
data = get_double_pendulum_chaotic_data(i=0, n=-1)

# ╔═╡ 9c8b3f8a-1b85-4c32-a416-ead51b244b94
begin
	anim = @animate for i in 1:1000
		scatter(legend=false, xlim=(0, 2500), ylim=(0, 2500))
		scatter!(data[[1, 3, 5], i], data[[2, 4, 6], i], color=[:red, :green, :blue])
		annotate!(250, 250, text("i=$i", :left))
	end

	gif(anim, fps=10)
end

# ╔═╡ 4a1ebdfe-2524-4d3e-b4ab-520af315063f


# ╔═╡ Cell order:
# ╟─194baef2-0417-11ec-05ab-4527ef614024
# ╠═38c9ced5-dcf8-4e03-ac07-7c435687861b
# ╠═5268feee-bda2-4612-9d4c-a1db424a11c7
# ╠═9c8b3f8a-1b85-4c32-a416-ead51b244b94
# ╠═4a1ebdfe-2524-4d3e-b4ab-520af315063f
