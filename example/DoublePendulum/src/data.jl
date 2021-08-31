using DataDeps
using CSV
using DataFrames

function register_double_pendulum_chaotic()
    register(DataDep(
        "DoublePendulumChaotic",
        """
        Dataset was generated on the basis of 21 individual runs of a double pendulum.
        Each of the recorded sequences lasted around 40s and consisted of around 17500 frames.

        * `x_red`: Horizontal pixel coordinate of the red point (the central pivot to the first pendulum)
        * `y_red`: Vertical pixel coordinate of the red point (the central pivot to the first pendulum)
        * `x_green`: Horizontal pixel coordinate of the green point (the first pendulum)
        * `y_green`: Vertical pixel coordinate of the green point (the first pendulum)
        * `x_blue`: Horizontal pixel coordinate of the blue point (the second pendulum)
        * `y_blue`: Vertical pixel coordinate of the blue point (the second pendulum)

        Page: https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/
        """,
        "https://dax-cdn.cdn.appdomain.cloud/dax-double-pendulum-chaotic/2.0.1/double-pendulum-chaotic.tar.gz",
        "4ca743b4b783094693d313ebedc2e8e53cf29821ee8b20abd99f8fb4c0866f8d",
        post_fetch_method=unpack
    ))
end

function get_data(; i=0, n=-1)
    data_path = joinpath(datadep"DoublePendulumChaotic", "original", "dpc_dataset_csv")
    df = CSV.read(
        joinpath(data_path, "$i.csv"),
        DataFrame,
        header=[:x_red, :y_red, :x_green, :y_green, :x_blue, :y_blue]
    )
    data = (n < 0) ? collect(Matrix(df)') : collect(Matrix(df)')[:, 1:n]

    return Float32.(data)
end

function preprocess(𝐱; Δt=2, ratio=0.9)
    # move red point to (0, 0)
    xs_red, ys_red = 𝐱[1, :], 𝐱[2, :]
    𝐱[3, :] -= xs_red; 𝐱[5, :] -= xs_red
    𝐱[4, :] -= ys_red; 𝐱[6, :] -= ys_red

    # needs only green and blue points
    𝐱 = reshape(𝐱[3:6, 1:Δt:end], 1, 4, :)
    # velocity of green and blue points
    ∇𝐱 = 𝐱[:, :, 2:end] - 𝐱[:, :, 1:(end-1)]

    𝐱 = cat(𝐱[:, :, 1:(end-1)], ∇𝐱, dims=1)

    n_train, n_test = floor(Int, ratio*size(𝐱)[end]), floor(Int, (1-ratio)*size(𝐱)[end])

    𝐱_train, 𝐲_train = 𝐱[:, :, 1:(n_train-1)], 𝐱[:, :, 2:n_train]
    𝐱_test, 𝐲_test = 𝐱[:, :, (end-n_test+1):(end-1)], 𝐱[:, :, (end-n_test+2):end]

    return 𝐱_train, 𝐲_train, 𝐱_test, 𝐲_test
end

function get_dataloader(; n_file=20, Δt=2, ratio=0.9, batchsize=100)
    𝐱_train, 𝐲_train = Array{Float32}(undef, 2, 4, 0), Array{Float32}(undef, 2, 4, 0)
    𝐱_test, 𝐲_test = Array{Float32}(undef, 2, 4, 0), Array{Float32}(undef, 2, 4, 0)
    for i in 0:(n_file-1)
        𝐱_train_i, 𝐲_train_i, 𝐱_test_i, 𝐲_test_i = preprocess(get_data(i=i), Δt=Δt, ratio=ratio)

        𝐱_train, 𝐲_train = cat(𝐱_train, 𝐱_train_i, dims=3), cat(𝐲_train, 𝐲_train_i, dims=3)
        𝐱_test, 𝐲_test = cat(𝐱_test, 𝐱_test_i, dims=3), cat(𝐲_test, 𝐲_test_i, dims=3)
    end

    loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=batchsize, shuffle=true)
    loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end
