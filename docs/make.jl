using imageGenerationWithDiffusionModels
using Documenter

DocMeta.setdocmeta!(imageGenerationWithDiffusionModels, :DocTestSetup, :(using imageGenerationWithDiffusionModels); recursive=true)

makedocs(;
    modules=[imageGenerationWithDiffusionModels],
    authors="Kajetan Andrzejak <andrzejak@campus.tu-berlin.de>, Eylul Bektur <bektur@campus.tu-berlin.de>, Maxim Häffner <m.haefner@campus.tu-berlin.de>, Maria Krzywnicka <maria.krzywnicka@campus.tu-berlin.de>",
    sitename="imageGenerationWithDiffusionModels.jl",
    format=Documenter.HTML(;
        canonical="https://kajetana.github.io/imageGenerationWithDiffusionModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kajetana/imageGenerationWithDiffusionModels.jl",
    devbranch="main",
)
