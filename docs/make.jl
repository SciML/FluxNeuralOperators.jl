using NeuralOperators
using Documenter

DocMeta.setdocmeta!(NeuralOperators, :DocTestSetup, :(using NeuralOperators); recursive=true)

makedocs(;
    modules=[NeuralOperators],
    authors="JingYu Ning <foldfelis@gmail.com> and contributors",
    repo="https://github.com/foldfelis/NeuralOperators.jl/blob/{commit}{path}#{line}",
    sitename="NeuralOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://foldfelis.github.io/NeuralOperators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/foldfelis/NeuralOperators.jl",
)
