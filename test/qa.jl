using NeuralOperators, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(NeuralOperators)
    Aqua.test_ambiguities(NeuralOperators, recursive = false)
    Aqua.test_deps_compat(NeuralOperators)
    Aqua.test_piracies(NeuralOperators, broken = true)
    Aqua.test_project_extras(NeuralOperators)
    Aqua.test_stale_deps(NeuralOperators)
    Aqua.test_unbound_args(NeuralOperators)
    Aqua.test_undefined_exports(NeuralOperators)
end
