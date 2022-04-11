export
    FourierNeuralOperator,
    MarkovNeuralOperator

"""
    FourierNeuralOperator(;
        ch=(2, 64, 64, 64, 64, 64, 128, 1),
        modes=(16, ),
        σ=gelu
    )

Fourier neural operator learns a neural operator with Dirichlet kernel to form a Fourier transformation.
It performs Fourier transformation across infinite-dimensional function spaces and learns better than neural operator.
"""
function FourierNeuralOperator(;
    ch=(2, 64, 64, 64, 64, 64, 128, 1),
    modes=(16, ),
    σ=gelu
)
    Transform = FourierTransform

    return Chain(
        Dense(ch[1], ch[2]),
        OperatorKernel(ch[2]=>ch[3], modes, Transform, σ),
        OperatorKernel(ch[3]=>ch[4], modes, Transform, σ),
        OperatorKernel(ch[4]=>ch[5], modes, Transform, σ),
        OperatorKernel(ch[5]=>ch[6], modes, Transform),
        Dense(ch[6], ch[7], σ),
        Dense(ch[7], ch[8]),
    )
end

"""
    MarkovNeuralOperator(;
        ch=(1, 64, 64, 64, 64, 64, 1),
        modes=(24, 24),
        σ=gelu
    )

Markov neural operator learns a neural operator with Fourier operators.
With only one time step information of learning, it can predict the following few steps with low loss
by linking the operators into a Markov chain.
"""
function MarkovNeuralOperator(;
    ch=(1, 64, 64, 64, 64, 64, 1),
    modes=(24, 24),
    σ=gelu
)
    Transform = FourierTransform

    return Chain(
        Dense(ch[1], ch[2]),
        OperatorKernel(ch[2]=>ch[3], modes, Transform, σ),
        OperatorKernel(ch[3]=>ch[4], modes, Transform, σ),
        OperatorKernel(ch[4]=>ch[5], modes, Transform, σ),
        OperatorKernel(ch[5]=>ch[6], modes, Transform, σ),
        Dense(ch[6], ch[7]),
    )
end
