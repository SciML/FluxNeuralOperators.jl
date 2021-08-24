# NeuralOperators

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![doc dev badge]][doc dev link] | [![ci badge]][ci link] [![codecov badge]][codecov link] |

[doc dev badge]: https://img.shields.io/badge/docs-dev-blue.svg
[doc dev link]: https://foldfelis.github.io/NeuralOperators.jl//dev

[ci badge]: https://github.com/foldfelis/NeuralOperators.jl/actions/workflows/CI.yml/badge.svg
[ci link]: https://github.com/foldfelis/NeuralOperators.jl/actions/workflows/CI.yml
[codecov badge]: https://codecov.io/gh/foldfelis/NeuralOperators.jl/branch/master/graph/badge.svg?token=JQH3MP1Y9R
[codecov link]: https://codecov.io/gh/foldfelis/NeuralOperators.jl

Neural operator is a novel deep learning architecture. It learns a operator, which is a mapping
between infinite-dimensional function spaces. It can be used to resolve [partial differential equations (PDE)](https://en.wikipedia.org/wiki/Partial_differential_equation).
Instead of solving by finite element method, a PDE problem can be resolved by learning a neural network to learn an operator
mapping from infinite-dimensional space (u, t) to infinite-dimensional space f(u, t). Neural operator learns a continuous function
between two continuous function spaces. The kernel can be trained on different geometry, which is learned from a graph.

Fourier neural operator learns a neural operator with Dirichlet kernel to form a Fourier transformation. It performs Fourier transformation across infinite-dimensional function spaces and learns better than neural operator.

Currently, `FourierOperator` is provided in this work.

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

Or you can just call:

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

## Examples

PDE training examples are provided in `example` folder.

### One-dimensional Burgers' equation

[Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation) example can be found in `example/Burgers`.
Use following commend to train model:

```julia
$ julia --proj

julia> using Burgers; Burgers.train()
```

### Double Pendulum

```julia
$ julia --proj

julia> using DoublePendulum; DoublePendulum.train()
```

## Roadmap

- [x] `FourierOperator` layer
- [x] One-dimensional Burgers' equation example
- [ ] `MarkovOperator` layer
- [x] Double pendulum prediction example
- [ ] `NeuralOperator` layer
- [ ] Poisson equation example

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
  - [zongyi-li/fourier_neural_operator](https://github.com/zongyi-li/fourier_neural_operator)
- [Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)
  - [zongyi-li/graph-pde](https://github.com/zongyi-li/graph-pde)
