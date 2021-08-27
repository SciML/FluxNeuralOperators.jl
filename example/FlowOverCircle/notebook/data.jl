### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 194baef2-0417-11ec-05ab-4527ef614024
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 38c9ced5-dcf8-4e03-ac07-7c435687861b
using FlowOverCircle, Plots

# ╔═╡ 5268feee-bda2-4612-9d4c-a1db424a11c7
begin 
	n = 51
	data = FlowOverCircle.gen_data(LinRange(100, 100-1+n, n))
end;

# ╔═╡ 9b02b6a2-33c3-4ca6-bfba-0bd74b664830
begin
	anim = @animate for i in 1:size(data)[end]
		heatmap(data[1, :, :, i]', color=:coolwarm, clim=(-1.5, 1.5))
		scatter!([64], [63], markersize=45, color=:black, legend=false, ticks=false)
		annotate!(5, 5, text("i=$i", :left))
	end
	gif(anim, fps=5)
end

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
		heatmap(states[1, :, :, i]', color=:coolwarm, clim=(-1.5, 1.5))
		scatter!([64], [63], markersize=45, color=:black, legend=false, ticks=false)
		annotate!(5, 5, text("i=$i", :left))
	end
	gif(anim_model, fps=5)
end

# ╔═╡ Cell order:
# ╟─194baef2-0417-11ec-05ab-4527ef614024
# ╠═38c9ced5-dcf8-4e03-ac07-7c435687861b
# ╠═5268feee-bda2-4612-9d4c-a1db424a11c7
# ╟─9b02b6a2-33c3-4ca6-bfba-0bd74b664830
# ╠═fbc287b8-f232-4350-9948-2091908e5a30
# ╟─a0b5e94c-a839-4cc0-a325-1a4ac39fafbc
