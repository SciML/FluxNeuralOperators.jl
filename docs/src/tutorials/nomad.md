# Nonlinear Manifold Decoders for Operator Learning (NOMADs)

NOMADs are similar to DeepONets in the aspect that they can learn when the input and output
function spaces are defined on different domains. Their architecture is different and use
nonlinearity to the latent codes to obtain the operator approximation. The architecture
involves an approximator to encode the input function space, which is directly concatenated
with the input function coordinates, and passed into a decoder net to give the output
function at the given coordinate.

```math
\begin{align*}
u(y) \xrightarrow{\mathcal{A}} & \; \beta \\
& \quad \searrow\\
& \quad \quad \mathcal{G}_{\theta} u(y) = \mathcal{D}(\beta, y) \\
& \quad \nearrow \\
y
\end{align*}
```

## Usage

Let's try to learn the anti-derivative operator for

```math
u(x) = sin(\alpha x)
```

That is, we want to learn

```math
\mathcal{G} : u \rightarrow v \\
```

such that

```math
v(x) = \frac{du}{dx} \quad \forall \; x \in [0, 2\pi], \; \alpha \in [0.5, 1]
```

### Copy-pastable code

```@example nomad_tutorial
using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie

rng = Random.default_rng()

eval_points = 1
data_size = 128
dim_y = 1
m = 32

xrange = range(0, 2π; length=m) .|> Float32
u_data = zeros(Float32, m, data_size)
α = 0.5f0 .+ 0.5f0 .* rand(Float32, data_size)

y_data = rand(Float32, 1, eval_points, data_size) .* 2π
v_data = zeros(Float32, eval_points, data_size)
for i in 1:data_size
    u_data[:, i] .= sin.(α[i] .* xrange)
    v_data[:, i] .= -inv(α[i]) .* cos.(α[i] .* y_data[1, :, i])
end

nomad = NOMAD(Chain(Dense(m => 8, σ), Dense(8 => 8, σ), Dense(8 => 7)),
    Chain(Dense(8 => 4, σ), Dense(4 => 1)))

ps, st = Lux.setup(rng, nomad)
data = [((u_data, y_data), v_data)]

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
            tstate)
        push!(losses, loss)
    end
    return losses
end

losses = train!(nomad, ps, st, data; epochs=1000)

lines(losses)
```
