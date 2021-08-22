module DoublePendulum

using NeuralOperators

include("data.jl")

__init__() = register_double_pendulum_chaotic()

end
