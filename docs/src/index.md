```@meta
CurrentModule = NeuralOperators
```

# NeuralOperators

Documentation for [NeuralOperators](https://github.com/foldfelis/NeuralOperators.jl).

```@index
```

## Layers

### Spectral convolutional layer

```math
F(s) = \mathcal{F} \{ v(x) \} \\
F'(s) = g(F(s)) \\
v'(x) = \mathcal{F}^{-1} \{ F'(s) \}
```

where ``v(x)`` and ``v'(x)`` denotes input and output function, ``\mathcal{F} \{ \cdot \}``, ``\mathcal{F}^{-1} \{ \cdot \}`` are Fourier transform, inverse Fourier transform, respectively. Function ``g`` is a linear transform for lowering Fouier modes.

```@docs
SpectralConv
```

Reference: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

---

### Fourier operator layer

```math
v_{t+1}(x) = \sigma(W v_t(x) + \mathcal{K} \{ v_t(x) \} )
```

where ``v_t(x)`` is the input function for ``t``-th layer and ``\mathcal{K} \{ \cdot \}`` denotes spectral convolutional layer. Activation function ``\sigma`` can be arbitrary non-linear function.

```@docs
FourierOperator
```

Reference: [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
