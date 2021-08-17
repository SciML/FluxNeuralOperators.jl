export
    FourierNeuralOperator

function FourierNeuralOperator()
    modes = (16, )
    ch = 64 => 64
    σ = gelu

    return Chain(
        Dense(2, 64),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes),
        Dense(64, 128, σ),
        Dense(128, 1),
        flatten
    )
end
