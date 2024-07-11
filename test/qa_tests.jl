@testitem "doctests: Quality Assurance" tags=[:qa] begin
    using Documenter, NeuralOperators

    DocMeta.setdocmeta!(
        NeuralOperators, :DocTestSetup, :(using NeuralOperators, Random); recursive=true)
    doctest(NeuralOperators; manual=false)
end

@testitem "Aqua: Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(NeuralOperators)
end

@testitem "Explicit Imports: Quality Assurance" tags=[:qa] begin
    using ExplicitImports

    # Skip our own packages
    @test check_no_implicit_imports(NeuralOperators; skip=(Base, Core, Lux)) === nothing
    @test check_no_stale_explicit_imports(NeuralOperators) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralOperators) === nothing
end
