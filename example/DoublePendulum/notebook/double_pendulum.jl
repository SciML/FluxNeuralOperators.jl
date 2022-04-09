### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 194baef2-0417-11ec-05ab-4527ef614024
using Pkg; Pkg.develop(path=".."); Pkg.activate("..")

# ╔═╡ 38c9ced5-dcf8-4e03-ac07-7c435687861b
begin
	using DoublePendulum
	using Plots
end

# ╔═╡ 396b5d7a-a7a4-4f22-a87e-39b405e8d62a
md"
# Double Pendulum

JingYu Ning
"

# ╔═╡ 2a606ecf-acf0-41ad-9290-7569dbb22b5a
md"
The data is provided by [IBM](https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/)

> In this dataset, videos of the double pendulum were taken using a high-speed Phantom Miro EX2 camera. To make the extraction of the arm positions easier, a matte black background was used, and the three datums were marked with red, green and blue fiducial markers. The camera was placed at 2 meters from the pendulum, with the axis of the objective aligned with the first pendulum datum. The pendulum was launched by hand, and the camera was motion triggered. The dataset was generated on the basis of 21 individual runs of the pendulum. Each of the recorded sequences lasted around 40s and consisted of around 17500 frames.
"

# ╔═╡ 5268feee-bda2-4612-9d4c-a1db424a11c7
data_x, data_y = DoublePendulum.preprocess(DoublePendulum.get_data(i=20));

# ╔═╡ 4d0b08a4-8a54-41fd-997f-ad54d4c984cd
m = DoublePendulum.get_model();

# ╔═╡ ad6302b2-3d62-4a3f-b8bf-f69bab80c7a4
ground_truth_data = cat(
	[data_x[:, :, :, i] 
	for i in 1:size(data_x, 3):size(data_x)[end]]..., dims=3
)[1, :, :];

# ╔═╡ 794374ce-6674-481d-8a3b-04db0f32d233
begin 
	n = 5
	inferenced_data = data_x[:, :, :, 1:1]
	for i in 1:n
		inferenced_data = cat(
			inferenced_data, 
			m(inferenced_data[:, :, :, i:i]), 
			dims=4
		)
	end
	
	inferenced_data = cat(
		[inferenced_data[:, :, :, i] for i in 1:n]..., dims=3
	)[1, :, :]
end;

# ╔═╡ 9c8b3f8a-1b85-4c32-a416-ead51b244b94
begin
	c = [
		RGB([239, 71, 111]/255...),
		RGB([6, 214, 160]/255...),
		RGB([17, 138, 178]/255...)
	]
	xi, yi = [2, 4, 6], [1, 3, 5]

	anim = @animate for i in 1:size(inferenced_data)[end]
		i_data = [0, 0, inferenced_data[:, i]...]
		g_data = [0, 0, ground_truth_data[:, i]...]

		scatter(
			legend=false, ticks=false,
			xlim=(-1500, 1500), ylim=(-1500, 1500), size=(400, 350)
		)
		plot!(i_data[xi], i_data[yi], color=:black)
		scatter!(i_data[xi], i_data[yi], color=c, markersize=8)
		plot!(g_data[xi], g_data[yi], color=:gray)
		scatter!(g_data[xi], g_data[yi], color=c, markersize=4)
		
		if i ≤ 30
			annotate!(-1400, -1400, text("t=$i", :left, color=:black))
		else
			annotate!(-1400, -1400, text("t=$i", :left, color=:red))
		end
	end

	gif(anim)
end

# ╔═╡ Cell order:
# ╟─396b5d7a-a7a4-4f22-a87e-39b405e8d62a
# ╟─2a606ecf-acf0-41ad-9290-7569dbb22b5a
# ╟─194baef2-0417-11ec-05ab-4527ef614024
# ╠═38c9ced5-dcf8-4e03-ac07-7c435687861b
# ╠═5268feee-bda2-4612-9d4c-a1db424a11c7
# ╠═4d0b08a4-8a54-41fd-997f-ad54d4c984cd
# ╠═ad6302b2-3d62-4a3f-b8bf-f69bab80c7a4
# ╠═794374ce-6674-481d-8a3b-04db0f32d233
# ╟─9c8b3f8a-1b85-4c32-a416-ead51b244b94
