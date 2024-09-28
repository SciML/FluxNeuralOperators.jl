# DeepONets

DeepONets are another class of networks that learn the mapping between two function spaces
by encoding the input function space and the location of the output space. The latent code
of the input space is then projected on the location laten code to give the output. This
allows the network to learn the mapping between two functions defined on different spaces.

```math
\begin{align*}
u(y) \xrightarrow{\text{branch}} & \; b \\
& \quad \searrow\\
&\quad \quad \mathcal{G}_{\theta} u(y) = \sum_k b_k t_k \\
&  \quad \nearrow \\
y \; \; \xrightarrow{\text{trunk}} \; \; & t  
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

```@example deeponet_tutorial
using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie

rng = Random.default_rng()

eval_points = 1
data_size = 64
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

deeponet = DeepONet(
    Chain(Dense(m => 8, σ), Dense(8 => 8, σ), Dense(8 => 8, σ)),
    Chain(Dense(1 => 4, σ), Dense(4 => 8, σ))
)

ps, st = Lux.setup(rng, deeponet)
data = [((u_data, y_data), v_data)]

function train!(model, ps, st, data; epochs=10)
    losses = []
    tstate = Training.TrainState(model, ps, st, Adam(0.001f0))
    for _ in 1:epochs, (x, y) in data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
            tstate)
        push!(losses, loss)
    end
    return losses
end

losses = train!(deeponet, ps, st, data; epochs=1000)

lines(losses)
```
