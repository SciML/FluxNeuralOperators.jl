function train_don(; n=300, cuda=true, learning_rate=0.001, epochs=400)
    if cuda && has_cuda()
        @info "Training on GPU"
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    x, y = get_data_don(n=n)

    xtrain = x[1:280, :]'
    ytrain = y[1:280, :]

    xval = x[end-19:end, :]' |> device
    yval = y[end-19:end, :] |> device

    grid = collect(range(0, 1, length=1024)') |> device

    opt = ADAM(learning_rate)

    m = DeepONet((1024,1024,1024), (1,1024,1024), gelu, gelu) |> device
    
    loss(X, y, sensor) = Flux.Losses.mse(m(X, sensor), y)
    evalcb() = @show(loss(xval, yval, grid))

    data = [(xtrain, ytrain, grid)] |> device
    Flux.@epochs epochs Flux.train!(loss, params(m), data, opt, cb=evalcb)
    ỹ = m(xval |> device, grid |> device)

    diffvec = vec(abs.(cpu(yval) .- cpu(ỹ)))
    mean_diff = sum(diffvec)/length(diffvec)
    return mean_diff
end
