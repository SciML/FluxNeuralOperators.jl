using CairoMakie

# Load the common utilities via Revise.jl if available
if @isdefined(includet)
    includet("../common.jl")
else
    include("../common.jl")
end

function train_burgers(; seed=1234, dataset="Burgers_R10", model_type=:fno)
    if model_type == :fno
        model = FourierNeuralOperator(; chs=(2, 64, 64, 64, 64, 64, 128, 1), modes=(16,),
            σ=gelu)
    else
        error("Unknown model type: $model_type")
    end

    trainloader, testloader = get_dataset(dataset; batchsize=512)

    ps, st = Lux.setup(Xoshiro(seed), model)

    opt = OptimiserChain(WeightDecay(1.0f-4), Adam(0.001f0))

    model, ps, st = train!(model, ps, st, trainloader, testloader, opt; epochs=100)

    return model, ps, st
end

model, ps, st = train_burgers()
x_data, y_data = get_dataset("Burgers_R10"; no_dataloader=Val(true));
st_ = Lux.testmode(st)
pred = first(model(x_data, ps, st_))

fig = with_theme(theme_latexfonts()) do
    fig = Figure(; size=(800, 800))

    for i in 1:2, j in 1:2
        idx = (i - 1) * 2 + j
        ax = Axis(fig[i, j]; xlabel=L"x", ylabel=L"u(x, t_{end})")

        l1 = lines!(ax, x_data[:, 1, idx], y_data[:, 1, idx]; linewidth=3)
        l2 = lines!(ax, x_data[:, 1, idx], pred[:, 1, idx]; linewidth=3, linsestyle=:dot,
            color=:red)

        if i == 1 && j == 1
            axislegend(ax, [l1, l2], ["Ground Truth", "Prediction"])
        end
    end

    return fig
end
