```@meta
CurrentModule = NeuralOperators
```

# NeuralOperators

Documentation for [NeuralOperators](https://github.com/foldfelis/NeuralOperators.jl).

| **Ground Truth** | **Inferenced** |
|:----------------:|:--------------:|
| ![](https://github.com/foldfelis/NeuralOperators.jl/blob/master/example/FlowOverCircle/gallery/ans.gif?raw=true) | ![](https://github.com/foldfelis/NeuralOperators.jl/blob/master/example/FlowOverCircle/gallery/inferenced.gif?raw=true) |

The demonstration shown above is Navier-Stokes equation learned by the `MarkovNeuralOperator` with only one time step information.
Example can be found in [`example/FlowOverCircle`](https://github.com/foldfelis/NeuralOperators.jl/tree/master/example/FlowOverCircle).

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

**DeepONet operator** (Deep Operator Network) learns a neural operator with the help of two sub-neural net structures described as the branch and the trunk network.
The branch network is fed the initial conditions data, whereas the trunk is fed with the locations where the target(output) is evaluated from the corresponding initial conditions.
It is important that the output size of the branch and trunk subnets is same so that a dot product can be performed between them.

Currently, the `OperatorKernel` layer is provided in this work.
As for model, there are `FourierNeuralOperator` and `MarkovNeuralOperator` provided.
Please take a glance at them [here](apis.html#Models).

## Quick start

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia-repl
pkg> add NeuralOperators
```

## Usage

### Fourier Neural Operator

```julia
model = Chain(
    # lift (d + 1)-dimensional vector field to n-dimensional vector field
    # here, d == 1 and n == 64
    Dense(2, 64),
    # map each hidden representation to the next by integral kernel operator
    OperatorKernel(64=>64, (16, ), FourierTransform, gelu),
    OperatorKernel(64=>64, (16, ), FourierTransform, gelu),
    OperatorKernel(64=>64, (16, ), FourierTransform, gelu),
    OperatorKernel(64=>64, (16, ), FourierTransform),
    # project back to the scalar field of interest space
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

### DeepONet

```julia
# tuple of Ints for branch net architecture and then for trunk net,
# followed by activations for branch and trunk respectively
model = DeepONet((32, 64, 72), (24, 64, 72), σ, tanh)
```

Or specify branch and trunk as separate `Chain` from Flux and pass to `DeepONet`

```julia
branch = Chain(Dense(32, 64, σ), Dense(64, 72, σ))
trunk = Chain(Dense(24, 64, tanh), Dense(64, 72, tanh))
model = DeepONet(branch, trunk)
```

You can again specify loss, optimization and training parameters just as you would for a simple neural network with Flux.

```julia
loss(xtrain, ytrain, sensor) = Flux.Losses.mse(model(xtrain, sensor), ytrain)
evalcb() = @show(loss(xval, yval, grid))

learning_rate = 0.001
opt = ADAM(learning_rate)
parameters = params(model)
Flux.@epochs 400 Flux.train!(loss, parameters, [(xtrain, ytrain, grid)], opt, cb=evalcb)
```
A more complete example using DeepONet architecture to solve Burgers' equation can be found in the [examples](../../example/Burgers/src/Burgers_deeponet.jl)
