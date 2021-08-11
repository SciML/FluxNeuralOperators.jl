export 
    FNO

function FNO(; ch = 64 => 64, modes=16)
    σ = x -> @. log1p(exp(x))

    return Chain(
        Dense(2, 64, init=c_glorot_uniform),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes, σ),
        FourierOperator(ch, modes),
        Dense(64, 128, σ, init=c_glorot_uniform),
        Dense(128, 1, init=c_glorot_uniform),
        flatten
    )
end
