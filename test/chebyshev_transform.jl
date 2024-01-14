using NeuralOperators
using Zygote
ch = 6
batch = 7
𝐱 = rand(30, 40, 50, ch, batch)
t = ChebyshevTransform((3, 4, 5))
@test ndims(t) == 3
@test size(transform(t, 𝐱)) == (30, 40, 50, ch, batch)
@test size(truncate_modes(t, transform(t, 𝐱))) == (3, 4, 5, ch, batch)
@test size(inverse(t, truncate_modes(t, transform(t, 𝐱)), size(𝐱))) ==
      (3, 4, 5, ch, batch)
g = gradient(x -> sum(inverse(t, truncate_modes(t, transform(t, x)), size(𝐱))), 𝐱)
@test size(g[1]) == (30, 40, 50, ch, batch)
