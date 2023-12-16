using Fetch, DataDeps, MAT, MLUtils
using Lux, LuxNeuralOperators, LuxCUDA
using Optimisers, Zygote
using TimerOutputs, ProgressLogging, Random
import BSON: @save, @load

const gdev = gpu_device()
const cdev = cpu_device()

# Make DataLoader work with ProgressLogging
Base.size(d::DataLoader) = (length(d),)

function register_if_notpresent(regname::String, datadep)
    name = datadep.name
    name != regname && return
    haskey(DataDeps.registry, name) && return
    return register(datadep)
end

function register_dataset(dataset::String)
    register_if_notpresent(dataset,
        DataDep("Burgers_R10", """Burgers R10""",
            "https://drive.google.com/file/d/16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe/view?usp=sharing";
            fetch_method=gdownload, post_fetch_method=unpack))

    register_if_notpresent(dataset,
        DataDep("Burgers_V1000", """Burgers V1000""",
            "https://drive.google.com/file/d/1G9IW_2shmfgprPYISYt_YS8xa87p4atu/view?usp=sharing";
            fetch_method=gdownload, post_fetch_method=unpack))

    register_if_notpresent(dataset,
        DataDep("Burgers_V100", """Burgers V100""",
            "https://drive.google.com/file/d/1nzT0-Tu-LS2SoMUCcmO1qyjQd6WC9OdJ/view?usp=sharing";
            fetch_method=gdownload, post_fetch_method=unpack))

    return
end

function get_dataset(dataset::String; return_eltype::Type{T}=Float32, batchsize::Int = 128,
        ratio::AbstractFloat=0.9, no_dataloader::Val{DT} = Val(false)) where {T, DT}
    register_dataset(dataset)
    root = @datadep_str dataset

    if dataset == "Burgers_R10"
        n = 2048
        Δsamples = 2^3
        grid_size = div(2^13, Δsamples)

        file = matopen(joinpath(root, "burgers_data_R10.mat"))
        x_data = Matrix{T}(read(file, "a")[1:n, 1:Δsamples:end]')
        y_data = Matrix{T}(read(file, "u")[1:n, 1:Δsamples:end]')
        close(file)

        x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
        x_loc_data[1, :, :] = reshape(repeat(LinRange(0, 1, grid_size), n), (grid_size, n))
        x_loc_data[2, :, :] .= x_data

        x, y = x_loc_data, reshape(y_data, 1, :, n)
    else
        error("Not Implemented Dataset: $(dataset)")
    end

    DT && return x, y

    data_train, data_test = splitobs((x, y); at=ratio)

    trainloader = DataLoader(data_train; batchsize, shuffle=true)
    testloader = DataLoader(data_test; batchsize, shuffle=true)

    return trainloader, testloader
end

@inline function l₂_loss(x, y)
    feature_dims = 2:(ndims(y) - 1)

    loss = sum(sqrt, sum(abs2, x .- y; dims = feature_dims))
    y_norm = sum(sqrt, sum(abs2, y; dims = feature_dims))

    return loss / y_norm
end

@inline l₂_loss(m, ps, x, y) = l₂_loss(m(x, ps), y)

function train!(model, ps, st, trainloader, testloader, opt; epochs = 500)
    ps = ps |> gdev
    st = st |> gdev
    st_opt = Optimisers.setup(opt, ps)

    model2 = Lux.Experimental.StatefulLuxLayer(model, ps, st)

    @progress "Epochs" for epoch in 1:epochs
        @progress name="Training" for (i, (x, y)) in enumerate(trainloader)
            x = x |> gdev
            y = y |> gdev
            l, gs = Zygote.withgradient(l₂_loss, model2, ps, x, y)
            ∂ps = gs[2]
            mod1(i, 10) == 1 && @info "Epoch: $epoch, Iter: $i, Loss: $l"
            Optimisers.update!(st_opt, ps, ∂ps)
        end

        st_ = Lux.testmode(model2.st)
        model_inf = Lux.Experimental.StatefulLuxLayer(model, ps, st_)

        total_loss = 0.0
        total_data = 0
        @progress name="Inference" for (x, y) in testloader
            x = x |> gdev
            y = y |> gdev
            total_loss += l₂_loss(model_inf, ps, x, y)
            total_data += 1
        end

        @info "Epoch: $epoch, Loss: $(total_loss / total_data)"
    end

    return model, ps |> cdev, st |> cdev
end