using DataDeps
using Fetch
using MAT

export
    get_data,
    preprocess

register(DataDep(
    "BurgersR10",
    """
    Burgers' equation dataset from
    [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
    """,
    "https://drive.google.com/file/d/16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe/view?usp=sharing",
    "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd",
    fetch_method=gdownload,
    post_fetch_method=unpack
))

function get_data(n=1000, Δsamples=2^3, grid_size=div(2^13, Δsamples))
    file = matopen(joinpath(datadep"BurgersR10", "burgers_data_R10.mat"))
    x_data = read(file, "a")[1:n, 1:Δsamples:end]
    y_data = read(file, "u")[1:n, 1:Δsamples:end]
    close(file)

    grid = reshape(repeat(LinRange(0, 1, grid_size), n),(grid_size, n))'
    x_data = cat(x_data, grid, dims=3)

    return x_data, y_data
end

function preprocess()
end
