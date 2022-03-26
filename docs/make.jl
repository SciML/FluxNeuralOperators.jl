using NeuralOperators
using Documenter
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), sorting=:nyt)

DocMeta.setdocmeta!(NeuralOperators, :DocTestSetup, :(using NeuralOperators); recursive=true)

makedocs(
    bib,
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
        "Introduction" => "introduction.md",
        "APIs" => "apis.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/SciML/NeuralOperators.jl",
)
