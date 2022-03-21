function train_mno(; cuda=true, η=1f-3, λ=1f-4, epochs=50)
    # GPU config
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    @info "gen data... "
    @time loader_train, loader_test = get_dataloader()

    # build model
    m = Chain(
        Dense(1, 64),
        OperatorKernel(64=>64, (24, 24), gelu),
        OperatorKernel(64=>64, (24, 24), gelu),
        OperatorKernel(64=>64, (24, 24), gelu),
        OperatorKernel(64=>64, (24, 24), gelu),
        Dense(64, 1),
    ) |> device

    # optimizer
    opt = Flux.Optimiser(WeightDecay(λ), Flux.ADAM(η))
    
    # parameters
    ps = Flux.params(model)

    # training
    min_loss = Inf32
    train_steps = 0
    @info "Start Training, total $(epochs) epochs"
    for epoch = 1:epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader_train))

        for (𝐱, 𝐲) in loader_train
            grad = gradient(() -> loss(model, 𝐱 |> device, 𝐲 |> device), ps)
            Flux.Optimise.update!(opt, ps, grad)
            train_loss = loss(model, loader_train, device)
            test_loss = loss(model, loader_test, device)

            # progress meter
            next!(progress; showvalues=[
                (:train_loss, train_loss),
                (:test_loss, test_loss)
            ])

            if test_loss ≤ min_loss
                update_model!(joinpath(@__DIR__, "../model/model.jld2"), m)
            end

            train_steps += 1
        end
    end

    return m
end