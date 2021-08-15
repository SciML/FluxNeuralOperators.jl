export
    UnitGaussianNormalizer,
    encode,
    decode,
    get_burgers_data,
    get_darcy_flow_data

struct UnitGaussianNormalizer{T}
    mean::Array{T}
    std::Array{T}
    ϵ::T
end

function UnitGaussianNormalizer(𝐱; ϵ=1f-5)
    dims = 1:ndims(𝐱)-1

    return UnitGaussianNormalizer(mean(𝐱, dims=dims), StatsBase.std(𝐱, dims=dims), ϵ)
end

encode(n::UnitGaussianNormalizer, 𝐱::AbstractArray) = @. (𝐱-n.mean) / (n.std+n.ϵ)
decode(n::UnitGaussianNormalizer, 𝐱::AbstractArray) = @. 𝐱 * (n.std+n.ϵ) + n.mean


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

function register_darcy_flow()
    register(DataDep(
        "DarcyFlow",
        """
        Darcy flow dataset from
        [fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
        """,
        "https://drive.google.com/file/d/1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf/view?usp=sharing",
        "802825de9da7398407296c99ca9ceb2371c752f6a3bdd1801172e02ce19edda4",
        fetch_method=gdownload,
        post_fetch_method=unpack
    ))
end

function register_datasets()
    register_burgers()
    register_darcy_flow()
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

function get_darcy_flow_data(; n=1024, Δsamples=5, T=Float32, test_data=false)
    # size(training_data) == size(testing_data) == (1024, 421, 421)
    file = test_data ? "piececonst_r421_N1024_smooth2.mat" : "piececonst_r421_N1024_smooth1.mat"
    file = matopen(joinpath(datadep"DarcyFlow", file))
    x_data = T.(permutedims(read(file, "coeff")[1:n, 1:Δsamples:end, 1:Δsamples:end], (3, 2, 1)))
    y_data = T.(permutedims(read(file, "sol")[1:n, 1:Δsamples:end, 1:Δsamples:end], (3, 2, 1)))
    close(file)

    x_dims = pushfirst!([size(x_data)...], 1)
    y_dims = pushfirst!([size(y_data)...], 1)
    x_data, y_data = reshape(x_data, x_dims...), reshape(y_data, y_dims...)

    x_normalizer, y_normalizer = UnitGaussianNormalizer(x_data), UnitGaussianNormalizer(y_data)

    return encode(x_normalizer, x_data), encode(y_normalizer, y_data), x_normalizer, y_normalizer
end
