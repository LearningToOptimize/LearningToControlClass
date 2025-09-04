using Documenter
using Pluto

include("definitions.jl")

repo_dir = dirname(@__DIR__)
build_dir = joinpath(repo_dir, "docs", "build")

plutos = [
    joinpath(repo_dir, "class01", "background_materials", "math_basics.jl"),
    joinpath(repo_dir, "class01", "background_materials", "optimization_basics.jl"),
    joinpath(repo_dir, "class01", "background_materials", "optimization_motivation.jl"),
    joinpath(repo_dir, "class01", "class01_intro.jl"),
]

if !isdir(build_dir)
    symlink(joinpath(repo_dir, "class01"),
        joinpath(repo_dir, "docs", "src", "class01")
    )
end

makedocs(
    sitename = "LearningToControlClass",
    format = Documenter.HTML(;
        assets = ["assets/wider.css", "assets/redlinks.css"],
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "macros" => make_macros_dict("docs/src/assets/definitions.tex"),
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
            ),
        )),
    ),
    pages  = [
        "Home"   => "index.md",
        "Class 1" => ["class01/class01.md",
            "class01/background_materials/git_adventure_guide.md",
        ],
    ],
)

rm(joinpath(repo_dir, "docs", "src", "class01"), force=true)

s = Pluto.ServerSession();
for pluto in plutos
    nb = Pluto.SessionActions.open(s, pluto; run_async=false)
    html_contents = Pluto.generate_html(nb; binder_url_js="undefined")
    filename = replace(pluto, repo_dir => build_dir)
    html_path = replace(filename, r"\.jl$" => ".html")
    mkpath(dirname(html_path))
    open(html_path, "w") do f
        write(f, html_contents)
    end
end

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
include("patch.jl")
deploydocs(
    repo="github.com/LearningToOptimize/LearningToControlClass.git",
    push_preview=true,
    deploy_config=L2CCGitHubActionsAllowExternalPreviews()
)
