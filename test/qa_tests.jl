@testitem "doctests: Quality Assurance" tags=[:qa] begin
    using Documenter, LuxNeuralOperators

    DocMeta.setdocmeta!(
        LuxNeuralOperators, :DocTestSetup, :(using LuxNeuralOperators); recursive=true)
    doctest(LuxNeuralOperators; manual=false)
end

@testitem "Aqua: Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(LuxNeuralOperators)
end

@testitem "Explicit Imports: Quality Assurance" tags=[:qa] begin
    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(LuxNeuralOperators; skip=(Base, Core, Lux)) === nothing
    @test check_no_stale_explicit_imports(LuxNeuralOperators) === nothing
    @test check_all_qualified_accesses_via_owners(LuxNeuralOperators) === nothing
end