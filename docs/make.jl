using FastHistograms
using Documenter

DocMeta.setdocmeta!(FastHistograms, :DocTestSetup, :(using FastHistograms); recursive = true)

makedocs(;
    modules = [FastHistograms],
    authors = "Octogonapus <firey45@gmail.com> and contributors",
    repo = "https://github.com/Octogonapus/FastHistograms.jl/blob/{commit}{path}#{line}",
    sitename = "FastHistograms.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Octogonapus.github.io/FastHistograms.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/Octogonapus/FastHistograms.jl", devbranch = "main")
