module Burgers_FEM

using FEniCS

function run_fem(ν = 1 / 1000)
    # parameters
    s = 1024  # x
    steps = 200  # t

    DT = Constant(1 / steps)
    dt = 1 / steps

    mesh = UnitIntervalMesh(s)
    V = FunctionSpace(mesh, "CG", 1)

    bc = DirichletBC(V, 0.0, "on_boundary")

    u_init = Expression("x[0]", degree = 1)
    u = TrialFunction(V)
    u_old = FeFunction(V)
    v = TestFunction(V)

    u = interpolate(u_init, V)
    assign(u_old, u)

    f = Expression("0.0", degree = 0)

    F = (dot(u - u_old, v) / DT
         + ν * inner(grad(u), grad(v))
         + inner(u * directional_derivative(u, 0), v)
         -
         dot(f, v)) * dx

    us = Vector{Float64}[]
    t = 0.0
    for n in 1:steps
        t = t + dt
        nlvsolve(F, u, bc)
        push!(us, get_array(u))
        assign(u_old, u)
    end

    return us
end

end
