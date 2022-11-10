using NeuralOperators
using Documenter
using DocumenterCitations

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), sorting = :nyt)

DocMeta.setdocmeta!(NeuralOperators, :DocTestSetup, :(using NeuralOperators);
                    recursive = true)

include("pages.jl")

makedocs(bib,
         modules = [NeuralOperators],
         authors = "JingYu Ning <foldfelis@gmail.com> and contributors",
         repo = "https://github.com/SciML/NeuralOperators.jl/blob/{commit}{path}#{line}",
         sitename = "NeuralOperators.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://docs.sciml.ai/NeuralOperators/stable/",
                                  assets = ["assets/favicon.ico"]),
         pages = pages)

deploydocs(;
           repo = "github.com/SciML/NeuralOperators.jl")
