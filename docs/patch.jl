struct L2CCGitHubActionsAllowExternalPreviews <: Documenter.GithubActions
    github_repository::String
    github_event_name::String
    github_ref::String
    github_triggering_actor::String
end
function L2CCGitHubActionsAllowExternalPreviews()
    github_repository = get(ENV, "GITHUB_REPOSITORY", "") # "JuliaDocs/Documenter.jl"
    github_event_name = get(ENV, "GITHUB_EVENT_NAME", "") # "push", "pull_request" or "cron" (?)
    github_ref = get(ENV, "GITHUB_REF", "") # "refs/heads/$(branchname)" for branch, "refs/tags/$(tagname)" for tags
    github_triggering_actor = get(ENV, "GITHUB_TRIGGERING_ACTOR", "")
    return L2CCGitHubActionsAllowExternalPreviews(github_repository, github_event_name, github_ref,
        github_triggering_actor)
end

function Documenter.deploy_folder(
        cfg::L2CCGitHubActionsAllowExternalPreviews;
        repo,
        repo_previews = nothing,
        deploy_repo = nothing,
        branch = "gh-pages",
        branch_previews = branch,
        devbranch,
        push_preview,
        devurl,
        tag_prefix = "",
        kwargs...
    )
    io = IOBuffer()
    all_ok = true
    ## Determine build type
    if cfg.github_event_name == "pull_request"
        build_type = :preview
    elseif occursin(r"^refs\/tags\/(.*)$", cfg.github_ref)
        build_type = :release
    else
        build_type = :devbranch
    end
    println(io, "Deployment criteria for deploying $(build_type) build from GitHub Actions:")
    ## The deploydocs' repo should match GITHUB_REPOSITORY
    repo_ok = occursin(cfg.github_repository, repo)
    all_ok &= repo_ok
    println(io, "- $(marker(repo_ok)) ENV[\"GITHUB_REPOSITORY\"]=\"$(cfg.github_repository)\" occurs in repo=\"$(repo)\"")
    if build_type === :release
        ## Do not deploy for PRs
        event_ok = in(cfg.github_event_name, ["push", "workflow_dispatch", "schedule", "release"])
        all_ok &= event_ok
        println(io, "- $(marker(event_ok)) ENV[\"GITHUB_EVENT_NAME\"]=\"$(cfg.github_event_name)\" is \"push\", \"workflow_dispatch\", \"schedule\" or \"release\"")
        ## If a tag exist it should be a valid VersionNumber
        m = match(r"^refs\/tags\/(.*)$", cfg.github_ref)
        tag_nobuild = version_tag_strip_build(m.captures[1]; tag_prefix)
        tag_ok = tag_nobuild !== nothing
        all_ok &= tag_ok
        println(io, "- $(marker(tag_ok)) ENV[\"GITHUB_REF\"]=\"$(cfg.github_ref)\" contains a valid VersionNumber")
        deploy_branch = branch
        deploy_repo = something(deploy_repo, repo)
        is_preview = false
        ## Deploy to folder according to the tag
        subfolder = m === nothing ? nothing : tag_nobuild
    elseif build_type === :devbranch
        ## Do not deploy for PRs
        event_ok = in(cfg.github_event_name, ["push", "workflow_dispatch", "schedule"])
        all_ok &= event_ok
        println(io, "- $(marker(event_ok)) ENV[\"GITHUB_EVENT_NAME\"]=\"$(cfg.github_event_name)\" is \"push\", \"workflow_dispatch\" or \"schedule\"")
        ## deploydocs' devbranch should match the current branch
        m = match(r"^refs\/heads\/(.*)$", cfg.github_ref)
        branch_ok = m === nothing ? false : String(m.captures[1]) == devbranch
        all_ok &= branch_ok
        println(io, "- $(marker(branch_ok)) ENV[\"GITHUB_REF\"] matches devbranch=\"$(devbranch)\"")
        deploy_branch = branch
        deploy_repo = something(deploy_repo, repo)
        is_preview = false
        ## Deploy to deploydocs devurl kwarg
        subfolder = devurl
    else # build_type === :preview
        m = match(r"refs\/pull\/(\d+)\/merge", cfg.github_ref)
        pr_number = tryparse(Int, m === nothing ? "" : m.captures[1])
        pr_ok = pr_number !== nothing
        all_ok &= pr_ok
        println(io, "- $(marker(pr_ok)) ENV[\"GITHUB_REF\"] corresponds to a PR number")
        ################################################################################
        ## Begin edits for L2CC
        ##
        ## Remove the PR origin checking:
        ##
        # if pr_ok
        #     pr_origin_matches_repo = verify_github_pull_repository(cfg.github_repository, pr_number)
        #     all_ok &= pr_origin_matches_repo
        #     println(io, "- $(marker(pr_origin_matches_repo)) PR originates from the same repository")
        # end
        ##
        ## End edits for L2CC
        ################################################################################
        btype_ok = push_preview
        all_ok &= btype_ok
        println(io, "- $(marker(btype_ok)) `push_preview` keyword argument to deploydocs is `true`")
        deploy_branch = branch_previews
        deploy_repo = something(repo_previews, deploy_repo, repo)
        is_preview = true
        ## deploydocs to previews/PR
        subfolder = "previews/PR$(something(pr_number, 0))"
    end
    ## GITHUB_ACTOR should exist (just check here and extract the value later)
    actor_ok = env_nonempty("GITHUB_ACTOR")
    all_ok &= actor_ok
    println(io, "- $(marker(actor_ok)) ENV[\"GITHUB_ACTOR\"] exists and is non-empty")
    ## GITHUB_TOKEN or DOCUMENTER_KEY should exist (just check here and extract the value later)
    token_ok = env_nonempty("GITHUB_TOKEN")
    key_ok = env_nonempty("DOCUMENTER_KEY")
    auth_ok = token_ok | key_ok
    all_ok &= auth_ok
    if key_ok
        println(io, "- $(marker(key_ok)) ENV[\"DOCUMENTER_KEY\"] exists and is non-empty")
    elseif token_ok
        println(io, "- $(marker(token_ok)) ENV[\"GITHUB_TOKEN\"] exists and is non-empty")
    else
        println(io, "- $(marker(auth_ok)) ENV[\"DOCUMENTER_KEY\"] or ENV[\"GITHUB_TOKEN\"] exists and is non-empty")
    end
    print(io, "Deploying: $(marker(all_ok))")
    @info String(take!(io))
    if build_type === :devbranch && !branch_ok && devbranch == "master" && cfg.github_ref == "refs/heads/main"
        @warn """
        Possible deploydocs() misconfiguration: main vs master
        Documenter's configured primary development branch (`devbranch`) is "master", but the
        current branch (from \$GITHUB_REF) is "main". This can happen because Documenter uses
        GitHub's old default primary branch name as the default value for `devbranch`.

        If your primary development branch is 'main', you must explicitly pass `devbranch = "main"`
        to deploydocs.

        See #1443 for more discussion: https://github.com/JuliaDocs/Documenter.jl/issues/1443
        """
    end
    if all_ok
        return DeployDecision(;
            all_ok = true,
            branch = deploy_branch,
            is_preview = is_preview,
            repo = deploy_repo,
            subfolder = subfolder
        )
    else
        return DeployDecision(; all_ok = false)
    end
end
