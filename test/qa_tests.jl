@testitem "doctests: Quality Assurance" tags=[:qa] begin
    using Documenter, NeuralOperators

    DocMeta.setdocmeta!(NeuralOperators, :DocTestSetup,
        :(using Lux, NeuralOperators, Random); recursive=true)
    doctest(NeuralOperators; manual=false)
end

@testitem "Aqua: Quality Assurance" tags=[:qa] begin
    using Aqua

    Aqua.test_all(NeuralOperators; ambiguities=false)
    Aqua.test_ambiguities(NeuralOperators; recursive=false)
end

@testitem "Explicit Imports: Quality Assurance" tags=[:qa] begin
    using ExplicitImports, Lux

    # Skip our own packages
    @test check_no_implicit_imports(NeuralOperators; skip=(Base, Core, Lux)) === nothing
    @test check_no_stale_explicit_imports(NeuralOperators) === nothing
    @test check_no_self_qualified_accesses(NeuralOperators) === nothing
    @test check_all_explicit_imports_via_owners(NeuralOperators) === nothing
    @test check_all_qualified_accesses_via_owners(NeuralOperators) === nothing
    @test_broken check_all_explicit_imports_are_public(NeuralOperators) === nothing  # mostly upstream problems
    @test_broken check_all_qualified_accesses_are_public(NeuralOperators) === nothing  # mostly upstream problems
end
