using NeuralOperators
using Documenter

DocMeta.setdocmeta!(NeuralOperators, :DocTestSetup, :(using NeuralOperators); recursive=true)

makedocs(;
    modules=[NeuralOperators],
    authors="JingYu Ning <foldfelis@gmail.com> and contributors",
    repo="https://github.com/SciML/NeuralOperators.jl/blob/{commit}{path}#{line}",
    sitename="NeuralOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="http://neuraloperators.sciml.ai",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "APIs" => "apis.md"
    ],
)

deploydocs(;
    repo="github.com/SciML/NeuralOperators.jl",
)
