using Preferences

Preferences.set_preferences!("LuxLib", "instability_check" => "error")
Preferences.set_preferences!("LuxCore", "instability_check" => "error")

using ReTestItems, Pkg, Test, InteractiveUtils, Hwloc, NeuralOperators

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

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 16))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@testset "NeuralOperators.jl Tests" begin
    ReTestItems.runtests(NeuralOperators; nworkers=RETESTITEMS_NWORKERS,
        nworker_threads=RETESTITEMS_NWORKER_THREADS, testitem_timeout=3600)
end
