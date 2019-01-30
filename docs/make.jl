using Documenter, RegionCovarianceDescriptor

makedocs(;
    modules=[RegionCovarianceDescriptor],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/rjww/RegionCovarianceDescriptor.jl/blob/{commit}{path}#L{line}",
    sitename="RegionCovarianceDescriptor.jl",
    authors="Robert Woods",
    assets=[],
)

deploydocs(;
    repo="github.com/rjww/RegionCovarianceDescriptor.jl",
)
