export
    FourierNeuralOperator

function FourierNeuralOperator(;
    ch=(2, 64, 64, 64, 64, 64, 128, 1),
    modes=(16, ),
    σ=gelu
)
    return Chain(
        Dense(ch[1], ch[2]),
        FourierOperator(ch[2]=>ch[3], modes, σ),
        FourierOperator(ch[3]=>ch[4], modes, σ),
        FourierOperator(ch[4]=>ch[5], modes, σ),
        FourierOperator(ch[5]=>ch[6], modes),
        Dense(ch[6], ch[7], σ),
        Dense(ch[7], ch[8]),
        flatten
    )
end
