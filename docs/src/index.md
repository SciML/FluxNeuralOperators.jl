```@meta
CurrentModule = NeuralOperators
```

# NeuralOperators

Documentation for [NeuralOperators](https://github.com/foldfelis/NeuralOperators.jl).

| **Ground Truth** | **Inferenced** |
|:----------------:|:--------------:|
| ![](https://github.com/foldfelis/NeuralOperators.jl/blob/master/example/FlowOverCircle/gallery/ans.gif?raw=true) | ![](https://github.com/foldfelis/NeuralOperators.jl/blob/master/example/FlowOverCircle/gallery/inferenced.gif?raw=true) |

The demonstration showing above is Navier-Stokes equation learned by the `MarkovNeuralOperator` with only one time step information.
Example can be found in [`example/FlowOverCircle`](https://github.com/foldfelis/NeuralOperators.jl/tree/master/example/FlowOverCircle).
The result is also provided [here](assets/notebook/mno.jl.html)

## Abstract

Neural operator is a novel deep learning architecture.
It learns a operator, which is a mapping between infinite-dimensional function spaces.
It can be used to resolve [partial differential equations (PDE)](https://en.wikipedia.org/wiki/Partial_differential_equation).
Instead of solving by finite element method, a PDE problem can be resolved by training a neural network to learn an operator mapping
from infinite-dimensional space (u, t) to infinite-dimensional space f(u, t).
Neural operator learns a continuous function between two continuous function spaces.
The kernel can be trained on different geometry, which is learned from a graph.

**Fourier neural operator** learns a neural operator with Dirichlet kernel to form a Fourier transformation.
It performs Fourier transformation across infinite-dimensional function spaces and learns better than neural operator.

**Markov neural operator** learns a neural operator with Fourier operators.
With only one time step information of learning, it can predict the following few steps with low loss
by linking the operators into a Markov chain.

Currently, the `FourierOperator` layer is provided in this work.
As for model, there are `FourierNeuralOperator` and `MarkovNeuralOperator` provided.
Please take a glance at them [here](apis.html#Models).

## Quick start

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia-repl
pkg> add NeuralOperators
```

## Usage

```julia
model = Chain(
    # project finite-dimensional data to infinite-dimensional space
    Dense(2, 64),
    # operator projects data between infinite-dimensional spaces
    FourierOperator(64=>64, (16, ), gelu),
    FourierOperator(64=>64, (16, ), gelu),
    FourierOperator(64=>64, (16, ), gelu),
    FourierOperator(64=>64, (16, )),
    # project infinite-dimensional function to finite-dimensional space
    Dense(64, 128, gelu),
    Dense(128, 1),
    flatten
)
```

Or one can just call:

```julia
model = FourierNeuralOperator(
    ch=(2, 64, 64, 64, 64, 64, 128, 1),
    modes=(16, ),
    σ=gelu
)
```

And then train as a Flux model.

```julia
loss(𝐱, 𝐲) = sum(abs2, 𝐲 .- model(𝐱)) / size(𝐱)[end]
opt = Flux.Optimiser(WeightDecay(1f-4), Flux.ADAM(1f-3))
Flux.@epochs 50 Flux.train!(loss, params(model), data, opt)
```
