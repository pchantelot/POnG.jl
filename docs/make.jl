using POnG
using Documenter

DocMeta.setdocmeta!(POnG, :DocTestSetup, :(using POnG); recursive=true)

makedocs(;
    modules=[POnG],
    authors="Pierre Chantelot",
    sitename="POnG.jl",
    format=Documenter.HTML(;
        canonical="https://pchantelot.github.io/POnG.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pchantelot/POnG.jl",
    devbranch="main",
)
