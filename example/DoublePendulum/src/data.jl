using DataDeps
using CSV
using DataFrames

export get_double_pendulum_chaotic_data

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

function get_double_pendulum_chaotic_data(; i=0, n=-1)
    data_path = joinpath(datadep"DoublePendulumChaotic", "original", "dpc_dataset_csv")
    df = CSV.read(
        joinpath(data_path, "$i.csv"),
        DataFrame,
        header=[:x_red, :y_red, :x_green, :y_green, :x_blue, :y_blue]
    )
    data = (n < 0) ? collect(Matrix(df)') : collect(Matrix(df)')[:, 1:n]

    data /= maximum(data)

    return Float32.(data)
end

function get_dataloader(; i=0, n_train=15001, n_test=1501, Δn=1024, batchsize=100)
    x = reshape(get_double_pendulum_chaotic_data(; i=i, n=-1), :)
    𝐱 = reshape(vcat([x[i:(i+6Δn-1)] for i in 1:6:(length(x)-6(Δn-1))]...), 6, 1024, :)

    𝐱_train, 𝐲_train = 𝐱[:, :, 1:(n_train-1)], 𝐱[:, :, 2:n_train]
    loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=batchsize, shuffle=true)

    𝐱_test, 𝐲_test = 𝐱[:, :, (end-n_test+1):(end-1)], 𝐱[:, :, (end-n_test+2):end]
    loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end
