using DataDeps
using Fetch
using MAT

function register_burgers()
    register(DataDep(
        "Burgers",
        """
        Burgers' equation dataset from
        [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
        """,
        "https://drive.google.com/file/d/17MYsKzxUQVaLMWodzPbffR8hhDHoadPp/view?usp=sharing",
        "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd",
        fetch_method=gdownload,
        post_fetch_method=unpack
    ))
end

function get_data(; n=2048, Δsamples=2^3, grid_size=div(2^13, Δsamples), T=Float32)
    file = matopen(joinpath(datadep"Burgers", "burgers_data_R10.mat"))
    x_data = T.(collect(read(file, "a")[1:n, 1:Δsamples:end]'))
    y_data = T.(collect(read(file, "u")[1:n, 1:Δsamples:end]'))
    close(file)

    x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
    x_loc_data[1, :, :] .= reshape(repeat(LinRange(0, 1, grid_size), n), (grid_size, n))
    x_loc_data[2, :, :] .= x_data

    return x_loc_data, y_data
end

function get_dataloader(; n_train=1800, n_test=200, batchsize=100)
    𝐱, 𝐲 = get_data(n=2048)

    𝐱_train, 𝐲_train = 𝐱[:, :, 1:n_train], 𝐲[:, 1:n_train]
    loader_train = Flux.DataLoader((𝐱_train, 𝐲_train), batchsize=batchsize, shuffle=true)

    𝐱_test, 𝐲_test = 𝐱[:, :, end-n_test+1:end], 𝐲[:, end-n_test+1:end]
    loader_test = Flux.DataLoader((𝐱_test, 𝐲_test), batchsize=batchsize, shuffle=false)

    return loader_train, loader_test
end
