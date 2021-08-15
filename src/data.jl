export
    get_burgers_data,
    get_navier_stokes

function register_burgers()
    register(DataDep(
        "Burgers",
        """
        Burgers' equation dataset from
        [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
        """,
        "https://drive.google.com/file/d/16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe/view?usp=sharing",
        "9cbbe5070556c777b1ba3bacd49da5c36ea8ed138ba51b6ee76a24b971066ecd",
        fetch_method=gdownload,
        post_fetch_method=unpack
    ))
end

# function register_navier_stokes()
#     register(DataDep(
#         "NavierStokes",
#         """
#         Navier–Stokes equations dataset from
#         [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
#         """,
#         "https://drive.google.com/file/d/1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d/view?usp=sharing",
#         "1a3b2893489dd1493923362bc74cd571e0b4f6ee290985eda060f4140df602d0",
#         fetch_method=gdownload,
#         post_fetch_method=unpack
#     ))
# end

function register_datasets()
    register_burgers()
    # register_navier_stokes()
end

function get_burgers_data(; n=2048, Δsamples=2^3, grid_size=div(2^13, Δsamples), T=Float32)
    file = matopen(joinpath(datadep"Burgers", "burgers_data_R10.mat"))
    x_data = T.(collect(read(file, "a")[1:n, 1:Δsamples:end]'))
    y_data = T.(collect(read(file, "u")[1:n, 1:Δsamples:end]'))
    close(file)

    x_loc_data = Array{T, 3}(undef, 2, grid_size, n)
    x_loc_data[1, :, :] .= reshape(repeat(LinRange(0, 1, grid_size), n), (grid_size, n))
    x_loc_data[2, :, :] .= x_data

    return x_loc_data, y_data
end

# function get_navier_stokes()
#     file = matopen(joinpath(datadep"NavierStokes", "ns_V1e-3_N5000_T50.mat"))
# end
