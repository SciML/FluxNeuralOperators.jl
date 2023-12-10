function FourierNeuralOperator(; chs = (2, 64, 64, 64, 64, 64, 128, 1),
        modes = (16,), σ = gelu, kwargs...)
    lifting = mapping_layers = []
    for i in 2:(length(chs) - 2)
        push!(mapping_layers, SpectralKernel(chs[i] => chs[i + 1], modes; σ))
    end
    mapping = Chain(mapping_layers...)
    project = Chain(Dense(chs[end - 2] => chs[end - 1], σ), Dense(chs[end - 1] => chs[end]))

    return Chain(;
        lifting = Dense(chs[1] => chs[2]),
        mapping = Chain([SpectralKernel(chs[i] => chs[i + 1], modes; σ, kwargs...)
                         for i in 2:(length(chs) - 3)]...),
        project = Chain(Dense(chs[end - 2] => chs[end - 1], σ),
            Dense(chs[end - 1] => chs[end])))
end
