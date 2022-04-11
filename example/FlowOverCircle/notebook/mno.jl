### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 194baef2-0417-11ec-05ab-4527ef614024
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 38c9ced5-dcf8-4e03-ac07-7c435687861b
using FlowOverCircle, Plots

# ╔═╡ 50ce80a3-a1e8-4ba9-a032-dad315bcb432
md"
# Markov Neural Operator

JingYu Ning
"

# ╔═╡ 59769504-ebd5-4c6f-981f-d03826d8e34a
md"
This demo trains a Markov neural operator (MNO) introduced by [Zongyi Li *et al.*](https://arxiv.org/abs/2106.06898) with only one time step information. Then composed the operator to a Markov chain and inference the Navier-Stokes equations."

# ╔═╡ 823b3547-6723-43cf-85e6-cc6eb44efea1
md"
## Generate data
"

# ╔═╡ 5268feee-bda2-4612-9d4c-a1db424a11c7
begin 
	n = 20
	data = FlowOverCircle.gen_data(LinRange(100, 100+n-1, n))
end;

# ╔═╡ 9b02b6a2-33c3-4ca6-bfba-0bd74b664830
begin
	anim = @animate for i in 1:size(data)[end]
		heatmap(data[1, 2:end-1, 2:end-1, i]', color=:coolwarm, clim=(-1.5, 1.5))
		scatter!(
			[size(data, 3)÷2-2.5], [size(data, 3)÷2-2.25], 
			markersize=40, color=:black, legend=false, ticks=false
		)
		annotate!(5, 5, text("i=$i", :left))
	end
	gif(anim, fps=2)
end

# ╔═╡ 55058635-c7e9-4ee3-81c2-0153e84f4c8e
md"
## Inference

Use the first data generated above as the initial state, and apply the operator recurrently.
"

# ╔═╡ fbc287b8-f232-4350-9948-2091908e5a30
begin
	m = FlowOverCircle.get_model()
	
	states = Array{Float32}(undef, size(data))
	states[:, :, :, 1] .= view(data, :, :, :, 1)
	for i in 2:size(data)[end]
		states[:, :, :, i:i] .= m(view(states, :, :, :, i-1:i-1))
	end
end

# ╔═╡ a0b5e94c-a839-4cc0-a325-1a4ac39fafbc
begin
	anim_model = @animate for i in 1:size(states)[end]
		heatmap(states[1, 2:end-1, 2:end-1, i]', color=:coolwarm, clim=(-1.5, 1.5))
		scatter!(
			[size(data, 3)÷2-2.5], [size(data, 3)÷2-2.25], 
			markersize=40, color=:black, legend=false, ticks=false
		)
		annotate!(5, 5, text("i=$i", :left))
	end
	gif(anim_model, fps=2)
end

# ╔═╡ Cell order:
# ╟─50ce80a3-a1e8-4ba9-a032-dad315bcb432
# ╟─59769504-ebd5-4c6f-981f-d03826d8e34a
# ╟─194baef2-0417-11ec-05ab-4527ef614024
# ╠═38c9ced5-dcf8-4e03-ac07-7c435687861b
# ╟─823b3547-6723-43cf-85e6-cc6eb44efea1
# ╠═5268feee-bda2-4612-9d4c-a1db424a11c7
# ╟─9b02b6a2-33c3-4ca6-bfba-0bd74b664830
# ╟─55058635-c7e9-4ee3-81c2-0153e84f4c8e
# ╠═fbc287b8-f232-4350-9948-2091908e5a30
# ╟─a0b5e94c-a839-4cc0-a325-1a4ac39fafbc
