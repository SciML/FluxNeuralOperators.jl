module DoublePendulum

using NeuralOperators
using Flux

include("data.jl")

__init__() = register_double_pendulum_chaotic()

end
