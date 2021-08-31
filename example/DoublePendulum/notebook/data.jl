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
data = DoublePendulum.get_data(i=0, n=-1)[:, end-2048+1:end]

# ╔═╡ 4d0b08a4-8a54-41fd-997f-ad54d4c984cd
m = DoublePendulum.get_model()

# ╔═╡ 794374ce-6674-481d-8a3b-04db0f32d233
begin
	n = 10
	
	ground_truth_data = 1 .- data[:, 1+n:1024+n]
	
	inferenced_data = m(reshape(data[:, 1:1024], 1, 6, :, 1))
	for i in 1:n
		inferenced_data = m(inferenced_data)
	end
	inferenced_data = 1 .- reshape(inferenced_data, 6, :)
end

# ╔═╡ 9c8b3f8a-1b85-4c32-a416-ead51b244b94
begin
	anim = @animate for i in 1:4:1024
		scatter(legend=false, xlim=(0, 1), ylim=(-0.5, 1), size=(600, 500))
		scatter!(
			inferenced_data[[2, 4, 6], i], inferenced_data[[1, 3, 5], i], 
			color=[
				RGB([239, 71, 111]/255...), 
				RGB([6, 214, 160]/255...), 
				RGB([17, 138, 178]/255...)
			],
			markersize=8
		)
		scatter!(
			ground_truth_data[[2, 4, 6], i], ground_truth_data[[1, 3, 5], i], 
			color=[
				RGB([255, 166, 158]/255...), 
				RGB([184, 242, 230]/255...), 
				RGB([174, 217, 224]/255...)
			],
			markersize=4
		)
		annotate!(0.1, -0.4, text("i=$i", :left))
	end

	gif(anim)
end

# ╔═╡ Cell order:
# ╟─194baef2-0417-11ec-05ab-4527ef614024
# ╠═38c9ced5-dcf8-4e03-ac07-7c435687861b
# ╠═5268feee-bda2-4612-9d4c-a1db424a11c7
# ╠═4d0b08a4-8a54-41fd-997f-ad54d4c984cd
# ╠═794374ce-6674-481d-8a3b-04db0f32d233
# ╟─9c8b3f8a-1b85-4c32-a416-ead51b244b94
