using ReTestItems, Pkg, ReTestItems, Test

const BACKEND_GROUP = lowercase(get(ENV, "BACKEND_GROUP", "all"))

const EXTRA_PKGS = String[]
(BACKEND_GROUP == "all" || BACKEND_GROUP == "cuda") && push!(EXTRA_PKGS, "LuxCUDA")
(BACKEND_GROUP == "all" || BACKEND_GROUP == "amdgpu") && push!(EXTRA_PKGS, "AMDGPU")

if !isempty(EXTRA_PKGS)
    @info "Installing Extra Packages for testing" EXTRA_PKGS=EXTRA_PKGS
    Pkg.add(EXTRA_PKGS)
    Pkg.update()
    Base.retry_load_extensions()
    Pkg.instantiate()
end

@testset "NeuralOperators.jl Tests" begin
    ReTestItems.runtests(@__DIR__)
end
