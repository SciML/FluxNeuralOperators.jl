function train_don()
    if has_cuda()
        @info "CUDA is on"
        device = gpu
        CUDA.allowscalar(false)
    else
        device = cpu
    end

    x, y = get_data_don(n=300)
    xtrain = x[1:280, :]' |> device
    xval = x[end-19:end, :]' |device

    ytrain = y[1:280, :] |> device
    yval = y[end-19:end, :] |> device

    grid = collect(range(0, 1, length=1024))' |> device

    learning_rate = 0.001
    opt = ADAM(learning_rate)

    m = DeepONet((1024,1024,1024),(1,1024,1024),gelu,gelu)
    loss(xtrain,ytrain,sensor) = Flux.Losses.mse(model(xtrain,sensor),ytrain)
    evalcb() = @show(loss(xval,yval,grid))

    Flux.@epochs 400 Flux.train!(loss, params(m), [(xtrain,ytrain,grid)], opt, cb = evalcb)
    ỹ = m(xval, grid)

    diffvec = vec(abs.((yval .- ỹ)))
    mean_diff = sum(diffvec)/length(diffvec)
    return mean_diff
end
