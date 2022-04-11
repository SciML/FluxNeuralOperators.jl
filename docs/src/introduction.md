# Introduction

Neural operator is a novel deep learning architecture.
It learns a operator, which is a mapping between infinite-dimensional function spaces.
It can be used to resolve [partial differential equations (PDE)](https://en.wikipedia.org/wiki/Partial_differential_equation).
Instead of solving by time-consuming finite element method, a PDE problem can be resolved by training a neural network to learn
an operator mapping from infinite-dimensional space ``(u, t)`` to infinite-dimensional space ``f(u, t)``.
Neural operator learns a continuous function between two continuous function spaces.
The kernel can be trained on different geometry, including regular Euclidean space or a graph topology.

## [Fourier Neural Operators](https://github.com/SciML/NeuralOperators.jl/blob/master/src/model.jl)

Fourier neural operator (FNO) learns a neural operator with Dirichlet kernel to form a Fourier transformation.
It performs Fourier transformation across infinite-dimensional function spaces and learns better than neural operator.

## [Markov Neural Operators](https://github.com/SciML/NeuralOperators.jl/blob/master/src/model.jl)

Markov neural operator (MNO) learns a neural operator with Fourier operators.
With only one time step information of learning, it can predict the following few steps with low loss
by linking the operators into a Markov chain.

## [Deep Operator Network](https://github.com/SciML/NeuralOperators.jl/blob/master/src/DeepONet.jl)

Deep operator network (DeepONet) learns a neural operator with the help of two sub-neural network structures described as the branch and the trunk network.
The branch network is fed the initial conditions data, whereas the trunk is fed with the locations where the target(output) is evaluated from the corresponding initial conditions.
It is important that the output size of the branch and trunk subnets is same so that a dot product can be performed between them.
