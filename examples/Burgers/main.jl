using CairoMakie

# Load the common utilities via Revise.jl if available
if @isdefined(includet)
    includet("../common.jl")
else
    include("../common.jl")
end

function train_burgers(; seed = 1234, dataset="Burgers_R10", model_type=:fno)
    if model_type == :fno
        model = FourierNeuralOperator(; chs=(2, 64, 64, 64, 64, 64, 128, 1), modes=(16,),
            σ=gelu)
    else
        error("Unknown model type: $model_type")
    end

    trainloader, testloader = get_dataset(dataset)

    ps, st = Lux.setup(Xoshiro(seed), model)

    opt = OptimiserChain(WeightDecay(1.0f-4), Adam(0.001f0))

    train!(model, ps, st, trainloader, testloader, opt; epochs = 500)

    return model, ps, st
end

model, ps, st = train_burgers()
x_data, y_data = get_dataset("Burgers_R10"; no_dataloader=Val(true));
pred = first(model(x_data, ps, st))

fig = with_theme(theme_latexfonts()) do
    fig = Figure(; size = (800, 800))

    for i in 1:2, j = 1:2
        idx = (i - 1) * 2 + j
        ax = Axis(fig[i, j]; xlabel=L"x", ylabel=L"u(x, t_{end})")

        l1 = lines!(ax, x_data[1, :, idx], y_data[1, :, idx]; linewidth=2)
        l2 = lines!(ax, x_data[1, :, idx], pred[1, :, idx]; linewidth=2)

        if i == 1 && j == 1
            axislegend(ax, [l1, l2], ["Ground Truth", "Prediction"])
        end
    end

    fig
end
