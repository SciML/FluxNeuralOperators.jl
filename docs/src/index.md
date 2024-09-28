# NeuralOperators

`NeuralOperators.jl` is a package written in Julia to provide the architectures for learning
mapping between function spaces, and learning grid invariant solution of PDEs.

## Installation

On Julia 1.10+, you can install `NeuralOperators.jl` by running

```julia
import Pkg
Pkg.add("NeuralOperators")
```

Currently provided operator architectures are :

  - [Fourier Neural Operators (FNOs)](tutorials/fno.md)
  - [DeepONets](tutorials/deeponet.md)
  - [Nonlinear Manifold Decoders for Operator Learning (NOMADs)](tutorials/nomad.md)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode=PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" *
                name *
                ".jl/tree/gh-pages/v" *
                version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" *
               name *
               ".jl/tree/gh-pages/v" *
               version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
