```@meta
CurrentModule = NeuralOperators
```

# NeuralOperators

| ![](https://github.com/foldfelis/NeuralOperators.jl/blob/main/example/FlowOverCircle/gallery/ans.gif?raw=true) | ![](https://github.com/foldfelis/NeuralOperators.jl/blob/main/example/FlowOverCircle/gallery/inferenced.gif?raw=true) |
|:----------------:|:--------------:|
| **Ground Truth** | **Inferenced** |

The demonstration shown above is Navier-Stokes equation learned by the `MarkovNeuralOperator` with only one time step information.
Example can be found in [`example/FlowOverCircle`](https://github.com/SciML/NeuralOperators.jl/tree/main/example/FlowOverCircle).

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
loss(𝐱, 𝐲) = l₂loss(model(𝐱), 𝐲)
opt = Flux.Optimiser(WeightDecay(1f-4), Flux.Adam(1f-3))
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
opt = Adam(learning_rate)
parameters = params(model)
Flux.@epochs 400 Flux.train!(loss, parameters, [(xtrain, ytrain, grid)], opt, cb=evalcb)
```
A more complete example using DeepONet architecture to solve Burgers' equation can be found in the [examples](https://github.com/SciML/NeuralOperators.jl/blob/main/example/Burgers/src/Burgers_deeponet.jl).

## Reproducibility
```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```