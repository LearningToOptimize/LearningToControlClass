### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ f0c826c7-b2e3-4dbf-b29d-37623aa4d7c6
begin
	import Pkg
	Pkg.activate(".")
	Pkg.status()
end

# ╔═╡ 054eb7cf-cb60-41a0-9a87-215e36dcf53d
begin
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
end

# ╔═╡ 7c9364ef-b0b3-4e2e-92e4-b7567a7e98e8
begin
	using ImageEdgeDetection
	using ImageFeatures, Clustering
	using Graphs
	using ColorTypes
	using Images, ImageIO
	Random.seed!(8803)
	function overlay(img_rgb::AbstractMatrix{<:Colorant},
                 hcs::AbstractVector,
                 vcs::AbstractVector,
                 verts::Dict{Tuple{Int,Int},Int};
                 dot_half::Int = 2)

	    ol = copy(img_rgb)
	
	    nrow, ncol = size(ol)
	
	    clampidx(i, hi) = max(1, min(hi, i))  
	
	    for y in hcs
	        r = clampidx(round(Int, y), nrow)
	        ol[r, :] .= RGB(1, 0, 0)
	    end
	
	    for x in vcs
	        c = clampidx(round(Int, x), ncol)
	        ol[:, c] .= RGB(0, 0, 1)
	    end
	
	    for (r_idx, c_idx) in keys(verts)
	        y = clampidx(round(Int, hcs[r_idx]), nrow)
	        x = clampidx(round(Int, vcs[c_idx]), ncol)
	
	        rrange = clampidx(y - dot_half, nrow):clampidx(y + dot_half, nrow)
	        crange = clampidx(x - dot_half, ncol):clampidx(x + dot_half, ncol)
	
	        ol[rrange, crange] .= RGB(0, 1, 0)
	    end
	
	    return ol
	end
	function cluster_coords(rhos; k::Int, tol = 4)
	    X = reshape(collect(rhos), 1, :)
	    R = kmeans(X, k; maxiter = 100, display = :none)
	
	    centres = sort(vec(R.centers))
	    uniq = Float64[]
	    for c in centres
	        if isempty(uniq) || abs(c - last(uniq)) > tol
	            push!(uniq, c)
	        end
	    end
	    return uniq
	end
	
	img  = load("layout.png")
	gimg = Float64.(gray.(Gray.(img)))
	
	_edges = detect_edges(
		gimg,
		Canny(spatial_scale = 1.2,
			  low  = ImageEdgeDetection.Percentile(10),
			  high = ImageEdgeDetection.Percentile(60))
	)
	edges_bool = _edges .> 0

	lines = hough_transform_standard(edges_bool;   
		 stepsize       = 1,
		 vote_threshold = 120,
		 max_linecount  = 300
	)
	h_rhos = [ρ for (ρ, θ) in lines if abs(sin(θ)) < 0.15]
	v_rhos = [ρ for (ρ, θ) in lines if abs(cos(θ)) < 0.15]
	
	h_cs = cluster_coords(h_rhos; k = 50)
	v_cs = cluster_coords(v_rhos; k = 25)

	is_floor(y,x) = gimg[Int.(clamp(round(y),1,size(gimg,1))),
                     Int.(clamp(round(x),1,size(gimg,2)))] > 0.8

	verts   = Dict{Tuple{Int,Int},Int}()
	vid     = 0
	n_rows = length(h_cs)
	n_cols = length(v_cs)
	
	for (ri,y) in enumerate(h_cs), (ci,x) in enumerate(v_cs)
		global vid
	    # skip the outer frame
		if !(ri == n_rows -5 && ci == 4)
		    if ri <= 27 || ri >= n_rows - 3 || ci <= 6 || ci >= n_cols - 3
		        continue                       
		    end
		end
	
	    if is_floor(y, x)
	        vid += 1
	        verts[(ri,ci)] = vid
	    end
	end

	g = SimpleGraph(vid)
	for ((r,c), v) in verts
	    if haskey(verts,(r+1,c)) add_edge!(g, v, verts[(r+1,c)]) end
	    if haskey(verts,(r,c+1)) add_edge!(g, v, verts[(r,c+1)]) end
	end
	@info "Vertices = $(nv(g))  |  Edges = $(ne(g))"

	add_edge!(g, 211, 212)

	overlay_img = overlay(img, h_cs, v_cs, verts) 
end

# ╔═╡ 6804ca32-8bd3-4dfc-ade6-f4239f28e9da
begin
	using SparseArrays
	Random.seed!(8803)
	function make_shopping_list(g::SimpleGraph,
	   verts::Dict{Tuple{Int,Int},Int};
	   n::Int = 8,
	   rng = Random.GLOBAL_RNG
  	)
	
	    @assert n ≤ nv(g) "n = $n is larger than number of vertices = $(nv(g))"

	    manhattan(a, b) = abs(a[1] - b[1]) + abs(a[2] - b[2])
	
	    rc_of = Dict(v => rc for (rc, v) in verts)
	
	    all_verts = collect(vertices(g))
	    sel       = Int[ rand(rng, all_verts) ]
	
	    while length(sel) < n
	        best_v, best_d = 0, -1
	        for v in all_verts
	            v in sel && continue
	            d = minimum( manhattan(rc_of[v], rc_of[s]) for s in sel )
	            if d > best_d
	                best_v, best_d = v, d
	            end
	        end
	        push!(sel, best_v)
	    end
	
	    sort(sel)
	end
	function overlay_items(img_rgb, hcs, vcs, verts, list, start_node=211, end_node=102; dot_half = 3, fade::Float64   = 0.40, gradc=false
	)
	    grey = RGB.(Gray.(img_rgb))
	    ol   = map(c -> RGB((1 - fade) * c.r + fade,
	                        (1 - fade) * c.g + fade,
	                        (1 - fade) * c.b + fade), grey)
	
	    nrow, ncol = size(ol)
	    clampidx(i, hi) = max(1, min(hi, i))
	
	    rc_of = Dict(v => rc for (rc, v) in verts)
	    function pix(v)
	        (r_idx, c_idx) = rc_of[v]
	        y = clampidx(round(Int, hcs[r_idx]), nrow)
	        x = clampidx(round(Int, vcs[c_idx]), ncol)
	        return y, x
	    end
	
	    for e in edges(g)
	        (y1, x1) = pix(src(e))
	        (y2, x2) = pix(dst(e))
	        n = max(abs(y2 - y1), abs(x2 - x1)) + 1
	        ys = round.(Int, range(y1, y2; length = n))
	        xs = round.(Int, range(x1, x2; length = n))
	        for (yy, xx) in zip(ys, xs)
	            ol[clampidx(yy, nrow), clampidx(xx, ncol)] = RGB(0.25,0.25,0.25)
	        end
	    end
	
	    for (r_idx, c_idx) in keys(verts)
	        y = clampidx(round(Int, hcs[r_idx]), nrow)
	        x = clampidx(round(Int, vcs[c_idx]), ncol)
	        rrange = clampidx(y - dot_half, nrow):clampidx(y + dot_half, nrow)
	        crange = clampidx(x - dot_half, ncol):clampidx(x + dot_half, ncol)
	        ol[rrange, crange] .= RGB(0, 0.7, 0)
	    end
		nlist = length(list)
		stp = 1/nlist
	    for (i, vid) in enumerate(list)
	        (y,x) = pix(vid)
	        rrange = clampidx(y - dot_half, nrow):clampidx(y + dot_half, nrow)
	        crange = clampidx(x - dot_half, ncol):clampidx(x + dot_half, ncol)
			if gradc
	        	ol[rrange, crange] .= RGB(0 + stp * (i-1), 0, 1 - stp * (i-1))
			else
				ol[rrange, crange] .= RGB(0,0,0)
			end
	    end
	
	    for (vid, col) in ((start_node, RGB(0,0,1)),
	                       (end_node,   RGB(1,0,0)))
	        (y,x) = pix(vid)
	        rrange = clampidx(y - dot_half, nrow):clampidx(y + dot_half, nrow)
	        crange = clampidx(x - dot_half, ncol):clampidx(x + dot_half, ncol)
	        ol[rrange, crange] .= col
	    end
	
	    return ol
	end

	plist = make_shopping_list(g, verts; n = 30)

	A_full = Float64.(adjacency_matrix(g))

	order = collect(plist)
	A_sub = A_full[order, order]
	overlay_img_items = overlay_items(img, h_cs, v_cs, verts, plist)
	overlay_img_items
end

# ╔═╡ 52281558-8396-471a-921f-f1185444c853
using JuMP, HiGHS

# ╔═╡ bcaf8412-964e-4d79-8db8-d69754fe4b83
using Unitful

# ╔═╡ 533cbe59-a206-418e-a397-b21c68e314da
md"
Author: Andrew Rosemberg

Date: 28 of July, 2025
"

# ╔═╡ 01c44cc2-68d5-11f0-2860-05c9ffbde13a
md"# Decisions Decisions: A Path to optimality

Once upon a time, a boy named **Pedro Paulo** 🤵 loved **Costco** supermarket 🛒. However, everytime there, he spendt more time ⏱️ and money 💸 than needed.

Let's help Pedro out!
"

# ╔═╡ 8d0fe751-aeb1-4ad1-a076-4c7bcd863a55
md"## Problem Setting

After some investigation, we got a hold of a real Costco layout! 🎉🎉 🗺️ 🎉🎉
"

# ╔═╡ 9eb11624-17db-438e-86e4-77b313da268b
md"Cool, now let's have a look at that shopping list 📝 and map things out!"

# ╔═╡ 75f6049d-cbdc-457f-9653-b498a21c30e6
@htl """

<img src="https://preview.redd.it/michelangelos-16th-century-grocery-list-he-illustrated-it-v0-wxtud7dbyphe1.png?width=640&crop=smart&auto=webp&s=f6a8dd56b75dc00d08ce519e4bb266523726246f" alt="Michelangelo's 16th century grocery list" width="300" height="200">

"""

# ╔═╡ 4108ad48-7be7-406e-b2ed-39888c7c559f
function itinerary_distance(full_itinerary, D)
	total_length = 0.0
	for i=1:length(full_itinerary)-1
		total_length += D[full_itinerary[i], full_itinerary[i+1]]
	end
	return total_length
end

# ╔═╡ 0bb8242e-3919-4551-94e6-8da74829a28a
function edge_distance_matrix(g::SimpleGraph,
                              verts::Dict{Tuple{Int,Int},Int},
                              h_cs::AbstractVector,
                              v_cs::AbstractVector;
                              scale::Real   = 1.0,
                              is_sparse::Bool  = true)

    n = nv(g)

    rc_of = Dict(v => rc for (rc,v) in verts)

    rows = Int[]; cols = Int[]; vals = Float64[]

    for e in edges(g)
        u, v = src(e), dst(e)

        (ru, cu) = rc_of[u]                
        (rv, cv) = rc_of[v]

        y1, x1 = h_cs[ru], v_cs[cu]      
        y2, x2 = h_cs[rv], v_cs[cv]

        d = scale * hypot(y2 - y1, x2 - x1)

        push!(rows, u); push!(cols, v); push!(vals, d)
        push!(rows, v); push!(cols, u); push!(vals, d)  
    end

    if is_sparse
        return sparse(rows, cols, vals, n, n)
    else
        D = zeros(Float64, n, n)
        for (r,c,v) in zip(rows, cols, vals)
            D[r,c] = v
        end
        return D
    end
end

# ╔═╡ 5f8709f0-b2e8-4f9e-8bf5-33485c64f891
begin
	D = edge_distance_matrix(g, verts, h_cs, v_cs; scale = 0.1)
	[A_full D]
end

# ╔═╡ 6cb27e4f-ad04-45f9-a6d5-fe8b1f79ed31
begin
	correct(md" ### Habemus Vehicle Routing Problem (VHR)!  🛒 👞 🛣️")
end

# ╔═╡ 70a32fbe-b2de-4400-9f16-91e0f186a7b1
md"""

> The vehicle routing problem is a combinatorial optimization and integer programming problem which asks `What is the optimal set of routes for a fleet of vehicles to traverse in order to deliver to a given set of customers?` The problem first appeared, as the truck dispatching problem, in a paper by George Dantzig and John Ramser in 1959. [^VRP]

"""

# ╔═╡ 7f868b8b-6cc9-47ce-9154-61ab325032e4
question_box(md"### Can't I just Dijkstra it?")

# ╔═╡ b14a113c-ba0d-4506-9a13-4c6d1a16bb3b
Foldable("What do you guys think?", md"""

> Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, a road network. It was conceived by computer scientist Edsger W. Dijkstra in 1956 and published three years later. [^Dijkstra]

So, not exactly, but let's see what we can do!

""")

# ╔═╡ d0b767ad-e1a2-4c4e-b988-1359aecde154
begin
	d = floyd_warshall_shortest_paths(g)     # or run Dijkstra from each vertex
	dist(u,v) = d.dists[u, v]               # distance lookup
end

# ╔═╡ 8705f186-8af4-4f6b-bb53-c3fa46d0b8ba
function greedy_tour(list, dist; start_node=211, end_node=102)
    tour = [start_node]
    remaining = Set([list[1:end]; end_node])
    while !isempty(remaining)
        last_v = tour[end]
        nxt = argmin(v -> dist(last_v, v), remaining)
        push!(tour, nxt)
        delete!(remaining, nxt)
    end
	push!(tour, end_node)
    return tour
end

# ╔═╡ 80e972a2-b9f2-4e78-a884-ded565d7ac96
function one_to_one_path(g::AbstractGraph, src::Integer, dst::Integer;
                         weights = nothing)

    ds = isnothing(weights) ?
            dijkstra_shortest_paths(g, src) :
            dijkstra_shortest_paths(g, weights, src)
    try
        return enumerate_paths(ds, dst)       
    catch err
        if err isa UndefVarError
            path = Int[]
            v = dst
            while v != 0 && v != src
                push!(path, v)
                v = ds.parents[v]
            end
            v == 0 && error("src and dst are disconnected")
            push!(path, src)
            return reverse(path)
        else
            rethrow(err)
        end
    end
end

# ╔═╡ 77e8077e-3059-4dc0-a41e-40ab31fa73d1
function stitch_paths(g::SimpleGraph, tour; weights = nothing)
    full = Int[]
    for (u, v) in zip(tour[1:end-1], tour[2:end])
        seg = one_to_one_path(g, u, v; weights)
        isempty(full) ? append!(full, seg) : append!(full, seg[2:end])
    end
    return full
end

# ╔═╡ 7b4f1d18-8a0b-4e7e-9493-aef1c9ea20e2
md"Well it works, but it is inefficient and ineffective"

# ╔═╡ 15e58149-6a7d-43f1-9c97-fb6ee102af05
begin
	hit_order = greedy_tour(plist, dist)
	full_itinerary = stitch_paths(g, hit_order)
	overlay_items(img,  h_cs, v_cs, verts, full_itinerary; gradc=true)
end

# ╔═╡ 8b2cdff7-bac4-4f74-a7b7-f10daaead623
begin
	total_length = itinerary_distance(full_itinerary, D)
	keep_working(md"The greedy approach took: $(round(total_length)) meters of distance! 🇫🇷🥖🗼")
end

# ╔═╡ 1d7942fa-dd53-4034-a822-78d7e76dbcd6
question_box(md"### How to model it as an integer programing problem?")

# ╔═╡ acb9d0fd-c024-44b7-b549-78875068050f
begin
	nverts=length(verts); nitems=length(plist);start_node=211;end_node=102;
	md"""
	#### What we have: 
	
	Adjecency Matrix, `A_full`, and the distance matrix, `D`.
	
	```math
	A_{\text{full}}[i,j] =
	\begin{cases}
	1 & \text{if node i is connected to j} \\
	0 & \text{if not}
	\end{cases}
	```
	
	```math
	D[i,j] =
	\begin{cases}
	\text{Euclidian distance from i to j} & \text{if connected} \\
	0 & \text{if not or i=j}
	\end{cases}
	```
	
	Number of vertices: `nverts=`$(nverts)
	
	Number of items on the list: `nitems=`$(nitems)

	Start node: $(start_node) | End node: $(end_node)
	
	
	`[A_full D]:`
	$([A_full D])
	
	Shopping list as a sequence of vertices: `plist=` 
	$(println(plist);"")
	
	"""
end

# ╔═╡ 083ddea0-f1db-46ef-b82c-5e10499bfb9d
aside(tip(md"Put your itinerary answer on `itinerary_answer`."))

# ╔═╡ 7c889414-b9c4-477d-8e57-79ee1518dc8c
begin
	#  BASIC SETS
	_V     = vertices(g)
	_A     = [(u,v) for e in edges(g) for (u,v) in ((src(e),dst(e)), (dst(e),src(e)))]
	_s, _e = 211, 102 						 # start, end
	_items = setdiff(Set(plist), [_s,_e])
	K      = length(_items) + 1              # units of flow we must deliver

	# MODEL
	model = Model(HiGHS.Optimizer)
	
	@variable(model, x[_A], Bin)
	@variable(model, 0 ≤ f[_A] ≤ K)
	
	out(v)  = sum(x[(v,w)]   for w in _V if (v,w) in _A)
	inn(v)  = sum(x[(u,v)]   for u in _V if (u,v) in _A)
	fout(v) = sum(f[(v,w)]   for w in _V if (v,w) in _A)
	finn(v) = sum(f[(u,v)]   for u in _V if (u,v) in _A)
	
	## degree / balance
	@constraint(model,  out(_s) - inn(_s) == 1)        # Costco Entrance
	@constraint(model,  inn(_e) - out(_e) == 1)        # Payment Exit
	@constraint(model, [i in _items],  inn(i) ≥ 1)     # Pick up items
	@constraint(model, [i in _items],  out(i) ≥ 1)
	@constraint(model, [v in setdiff(_V, [_s,_e])], inn(v) == out(v))
	
	## multi-unit single-commodity flow
	@constraint(model, [a in _A], f[a] ≤ K * x[a])     # deactivate on unused arcs
	
	@constraint(model, fout(_s) - finn(_s) ==  K)      # source injects K units
	@constraint(model, finn(_e) - fout(_e) ==  1)      # _e ‘consumes’ one unit
	@constraint(model, [i in _items],  finn(i) - fout(i) == 1)   # each item consumes 1
	@constraint(model, [v in setdiff(_V, [_s,_e] ∪ _items)],
	                      fout(v) == finn(v))          # pure transshipment elsewhere
	
	## objective
	@objective(model, Min, sum(D[u,v] * x[(u,v)] for (u,v) in _A))
	
	optimize!(model)
end

# ╔═╡ 8e507678-5d40-4a44-9aac-5701cc27f8ad
hint(md"
	 
1) It may be useful to create a binary variable that:

```math
A[i,j] =
\begin{cases}
1 & \text{if arc (i,j) is visited} \\
0 & \text{if not}
\end{cases}
```

2) Think of how we enforce entrance at `start_node` and exit `end_node`.

3) Need to visit all items at least once!

4) How you will reconstruct the path if you don't represent order?
	 
")

# ╔═╡ f7dc2333-d85f-44d7-bd5e-2f497a23f32b
function euler!(adj, start)
	st=[start]; path=Int[]
	while !isempty(st)
		u = st[end]
		if isempty(get(adj,u,Int[]))
			push!(path, pop!(st))
		else
			push!(st, pop!(adj[u]))           # consume arc once
		end
	end
	reverse(path)                             # ends at _e by construction
end

# ╔═╡ 7db1a83c-0100-4b6b-a97b-a7d59520dc22
begin
	adj = Dict{Int,Vector{Int}}()
	for (u,v) in _A
	    if value(x[(u,v)]) > 0.5
	        push!(get!(adj,u,Int[]), v)
	    end
	end
	itinerary_answer = euler!(deepcopy(adj), _s)
	@assert itinerary_answer[1] == _s && itinerary_answer[end] == _e
	itinerary_answer
end

# ╔═╡ 1c527fc1-f4fd-4a59-8f0b-63413b1bea56
# itinerary_answer = missing # replace with your answer

# ╔═╡ f5faca5f-abc3-49f1-add5-385de0ddb5b7
begin
	if ismissing(itinerary_answer)
		still_missing()
	else
		length_walk = itinerary_distance(itinerary_answer, D)
		if length_walk <= 312
			correct(md" $(round(length_walk)) meters. You have found the optimal path!")
		elseif length_walk <= total_length
			almost(md" $(round(length_walk)) meters. Nice you have beaten the greedy algorithm! But there is still room for improvement.")
		else
			keep_working(md" $(round(length_walk)) meters. You should at least beat the greedy algorithm. I thrust in you!")
		end
	end
end

# ╔═╡ 5de8429f-349d-4441-90b4-6f7caaf6b5e4
if !ismissing(itinerary_answer)
	overlay_items(img,  h_cs, v_cs, verts, full_itinerary; gradc=true)
end

# ╔═╡ e7cc612d-fc6a-4976-89b2-821064dc612c
still_missing()

# ╔═╡ e04cfe19-5827-4bf2-8183-a13bad579497
md"## Now Let's get a bit crazy!
"

# ╔═╡ 263969f2-5e0f-4d7b-9b42-eb033861e4e9
question_box(md"#### Would our solution change if we considered physics?")

# ╔═╡ bf94beb1-f5c9-45b0-a864-94f773a3c198
Foldable(md"#### How do we model the dynamics of a cart?", md"""

> The change of motion of an object is proportional to the force impressed; and is made in the direction of the straight line in which the force is impressed. [^Newton]

```math
F = m a
```

""")

# ╔═╡ 85f6ac91-bcc7-44de-948c-5a631a82c846
Foldable(md"#### What is an average force when pushing?", md"""

For general pushing, a force between 300 and 350 N is considered typical for a human. **Pedro is strong** and can push a cart with up to `350N`!

""")

# ╔═╡ c0cdf191-5350-4738-882f-8ded20168dbb
begin
	F = 350u"N"; m = 90u"kg"
	a = uconvert(u"m/s^2", F/m)
end

# ╔═╡ 6ab03a08-6cda-4f35-bd80-e0ace02b86b5
Foldable(md"#### Can we know the position of Pedro and his cart at any given moment?", md"""

```math
x[t+1] = x[t] + \dot{x}[t] * \Delta_t
```
```math
\dot{x}[t] = \dot{x}[t-1] + \ddot{x}
```

""")

# ╔═╡ e4baeff0-f380-4269-9818-9fc43c1f802d


# ╔═╡ f31d8852-ca07-46c9-bbea-5dd8f476c25c
	md"## References"

# ╔═╡ 178c6168-b515-4220-b37f-2c31b34e045c
begin
	MarkdownLiteral.@markdown(
"""
[^Dijkstra]: [Wikipedia on Dijkstra Algorithm]("https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#cite_note-Dijkstra19592-6")

[^VRP]:$(DOI("10.1016/j.ejor.2019.10.010"))

[^Newton]: [Wikipedia on Newton laws ofmotion]("https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion")

"""
)
end

# ╔═╡ 2e57bcd7-af6a-4813-8a2a-f7a71752a2de


# ╔═╡ Cell order:
# ╟─f0c826c7-b2e3-4dbf-b29d-37623aa4d7c6
# ╟─054eb7cf-cb60-41a0-9a87-215e36dcf53d
# ╟─533cbe59-a206-418e-a397-b21c68e314da
# ╟─01c44cc2-68d5-11f0-2860-05c9ffbde13a
# ╟─8d0fe751-aeb1-4ad1-a076-4c7bcd863a55
# ╟─7c9364ef-b0b3-4e2e-92e4-b7567a7e98e8
# ╟─9eb11624-17db-438e-86e4-77b313da268b
# ╟─75f6049d-cbdc-457f-9653-b498a21c30e6
# ╟─6804ca32-8bd3-4dfc-ade6-f4239f28e9da
# ╟─4108ad48-7be7-406e-b2ed-39888c7c559f
# ╟─0bb8242e-3919-4551-94e6-8da74829a28a
# ╠═5f8709f0-b2e8-4f9e-8bf5-33485c64f891
# ╟─6cb27e4f-ad04-45f9-a6d5-fe8b1f79ed31
# ╟─70a32fbe-b2de-4400-9f16-91e0f186a7b1
# ╟─7f868b8b-6cc9-47ce-9154-61ab325032e4
# ╟─b14a113c-ba0d-4506-9a13-4c6d1a16bb3b
# ╟─d0b767ad-e1a2-4c4e-b988-1359aecde154
# ╟─8705f186-8af4-4f6b-bb53-c3fa46d0b8ba
# ╟─80e972a2-b9f2-4e78-a884-ded565d7ac96
# ╟─77e8077e-3059-4dc0-a41e-40ab31fa73d1
# ╟─7b4f1d18-8a0b-4e7e-9493-aef1c9ea20e2
# ╟─15e58149-6a7d-43f1-9c97-fb6ee102af05
# ╟─8b2cdff7-bac4-4f74-a7b7-f10daaead623
# ╟─1d7942fa-dd53-4034-a822-78d7e76dbcd6
# ╠═52281558-8396-471a-921f-f1185444c853
# ╟─acb9d0fd-c024-44b7-b549-78875068050f
# ╟─083ddea0-f1db-46ef-b82c-5e10499bfb9d
# ╠═7c889414-b9c4-477d-8e57-79ee1518dc8c
# ╟─8e507678-5d40-4a44-9aac-5701cc27f8ad
# ╠═f7dc2333-d85f-44d7-bd5e-2f497a23f32b
# ╠═7db1a83c-0100-4b6b-a97b-a7d59520dc22
# ╠═1c527fc1-f4fd-4a59-8f0b-63413b1bea56
# ╟─f5faca5f-abc3-49f1-add5-385de0ddb5b7
# ╟─5de8429f-349d-4441-90b4-6f7caaf6b5e4
# ╠═e7cc612d-fc6a-4976-89b2-821064dc612c
# ╟─e04cfe19-5827-4bf2-8183-a13bad579497
# ╟─263969f2-5e0f-4d7b-9b42-eb033861e4e9
# ╟─bf94beb1-f5c9-45b0-a864-94f773a3c198
# ╟─85f6ac91-bcc7-44de-948c-5a631a82c846
# ╠═bcaf8412-964e-4d79-8db8-d69754fe4b83
# ╠═c0cdf191-5350-4738-882f-8ded20168dbb
# ╠═6ab03a08-6cda-4f35-bd80-e0ace02b86b5
# ╠═e4baeff0-f380-4269-9818-9fc43c1f802d
# ╟─f31d8852-ca07-46c9-bbea-5dd8f476c25c
# ╟─178c6168-b515-4220-b37f-2c31b34e045c
# ╠═2e57bcd7-af6a-4813-8a2a-f7a71752a2de
