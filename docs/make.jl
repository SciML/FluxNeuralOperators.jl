using Documenter, NeuralOperators

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml"; force=true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml"; force=true)

ENV["GKSwstype"] = "100"
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("pages.jl")

makedocs(;
    sitename="NeuralOperators.jl",
    clean=true,
    doctest=false,
    linkcheck=true,
    modules=[NeuralOperators],
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
        canonical="https://luxdl.github.io/NeuralOperators.jl/"),
    pages
)

deploydocs(; repo="github.com/LuxDL/NeuralOperators.jl.git", push_preview=true)
