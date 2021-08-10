module NeuralOperators
    function __init__()
        register_datasets()
    end

    include("preprocess.jl")
    include("fourier.jl")
end
