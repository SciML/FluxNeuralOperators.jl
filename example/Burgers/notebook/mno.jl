using Burgers, Plots
using DataDeps, MAT, MLUtils
using NeuralOperators, Flux
using CUDA, BSON

dataset = Burgers.get_data(n = 1000);

m = Burgers.get_model();
input_data, ground_truth = dataset[1], dataset[2];

i = 1
plot(input_data[1, :, 1], ground_truth[1, :, i], label = "ground_truth",
     title = "                                              Burgers equation u(x,T_end)");
p1 = plot!(input_data[1, :, 1], m(view(input_data, :, :, i:i))[1, :, 1], label = "predict");

plot(input_data[1, :, 1], ground_truth[1, :, i + 1], label = "ground_truth");
p2 = plot!(input_data[1, :, 1], m(view(input_data, :, :, (i + 1):(i + 1)))[1, :, 1],
           label = "predict");
i = 3

plot(input_data[1, :, 1], ground_truth[1, :, i], label = "ground_truth");
p3 = plot!(input_data[1, :, 1], m(view(input_data, :, :, i:i))[1, :, 1], label = "predict");

plot(input_data[1, :, 1], ground_truth[1, :, i + 1], label = "ground_truth");
p4 = plot!(input_data[1, :, 1], m(view(input_data, :, :, (i + 1):(i + 1)))[1, :, 1],
           label = "predict");
p = plot(p1, p2, p3, p4)
