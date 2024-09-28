# Fourier Neural Operators (FNOs)

FNOs are a subclass of Neural Operators that learn the learn the kernel $\Kappa_{\theta}$,
parameterized on $\theta$ between function spaces:

```math
(\Kappa_{\theta}u)(x) = \int_D \kappa_{\theta}(a(x), a(y), x, y) dy  \quad \forall x \in D
```

The kernel makes up a block $v_t(x)$ which passes the information to the next block as:

```math
v^{(t+1)}(x) = \sigma((W^{(t)}v^{(t)} + \Kappa^{(t)}v^{(t)})(x))
```

FNOs choose a specific kernel $\kappa(x,y) = \kappa(x-y)$, converting the kernel into a
convolution operation, which can be efficiently computed in the fourier domain.

```math
\begin{align*}
(\Kappa_{\theta}u)(x) 
&= \int_D \kappa_{\theta}(x - y) dy  \quad \forall x \in D\\
&= \mathcal{F}^{-1}(\mathcal{F}(\kappa_{\theta}) \mathcal{F}(u))(x) \quad \forall x \in D
\end{align*}
```

where $\mathcal{F}$ denotes the fourier transform. Usually, not all the modes in the
frequency domain are used with the higher modes often being truncated.

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

```@example fno_tutorial
using NeuralOperators, Lux, Random, Optimisers, Zygote, CairoMakie

rng = Random.default_rng()

data_size = 128
m = 32

xrange = range(0, 2π; length=m) .|> Float32;
u_data = zeros(Float32, m, 1, data_size);
α = 0.5f0 .+ 0.5f0 .* rand(Float32, data_size);
v_data = zeros(Float32, m, 1, data_size);

for i in 1:data_size
    u_data[:, 1, i] .= sin.(α[i] .* xrange)
    v_data[:, 1, i] .= -inv(α[i]) .* cos.(α[i] .* xrange)
end

fno = FourierNeuralOperator(gelu; chs=(1, 64, 64, 128, 1), modes=(16,), permuted=Val(true))

ps, st = Lux.setup(rng, fno);
data = [(u_data, v_data)];

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

losses = train!(fno, ps, st, data; epochs=100)

lines(losses)
```
