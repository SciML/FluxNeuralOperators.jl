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
end

res = train_burgers()
