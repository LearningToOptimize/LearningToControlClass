### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 462afff0-2ae3-4730-b48e-cf475fc9e14f
begin
	using Pkg
	Pkg.activate("class08/pluto_env")
	Pkg.instantiate()  # Installs packages from Manifest.toml if needed
	
	# Then import each package you want to use
	using Graphs
	using NetworkLayout
	using Plots
	using LinearAlgebra
	using PlutoUI
	using Statistics
	using GraphRecipes
	using Colors
	using PlutoTeachingTools
	using Random
end

# ╔═╡ 195683a4-b093-46af-9fb2-0a8a67d996e8
md"""
References:

* Olfati-Saber, Reza, J. Alex Fax, and Richard M. Murray. "Consensus and cooperation in networked multi-agent systems." Proceedings of the IEEE 95.1 (2007): 215-233.

* Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.

* Summers, Tyler H., and John Lygeros. "Distributed model predictive consensus via the alternating direction method of multipliers." 2012 50th annual Allerton conference on communication, control, and computing (Allerton). IEEE, 2012.

* Piansky, R., Stinchfield, G., Kody, A., Molzahn, D. K., & Watson, J. P. (2024). Long duration battery sizing, siting, and operation under wildfire risk using progressive hedging. Electric Power Systems Research, 235, 110785.
"""

# ╔═╡ 75bdf059-c8ac-4f9c-b023-3c010b4389cb
md"""
# Consensus, ADMM, and Distributed Optimal Control

## Learning Objectives

* Understand the intuition behind consensus and why it arises in distributed settings
* Learn ADMM and how it enables distributed optimization
* Explore applications to distributed control and model predictive control (MPC)

## Table of Contents

1. Centralized vs. Distributed Control
2. Consensus Algorithms
3. ADMM for Consensus Optimization
4. Distributed Control Application
5. Connections with Other Lectures
6. Summary and Discussion

"""

# ╔═╡ fe6b8381-edeb-4d0c-87f5-4ecd2b7b9183
md"""
## 1. Centralized vs. Distributed Control

Consider a cooperative control setting involving a network of autonomous vehicles.  
Each agent $i$ is described by a simple double-integrator model:

```math
\dot{x}_i = v_i, \quad \dot{v}_i = u_i.
```

Suppose a centralized controller aims to maintain a desired formation or platoon.
__Since each agent’s behavior depends on its neighbors’ states__, the overall coordination problem can be expressed as:

```math
\min_{u} \sum_{i\in V} \int_{0}^{T} \left( ||x_i - x_i^{\text{ref}}||^2 + ||u_i||^2 + \alpha\sum_{j \in \mathcal{N}_i}||x_i - x_j||^2 \right) dt
```

where $\mathcal{N}_i$ denotes the neighbors of agent $i$ within the communication graph $G = (V,E)$.

While centralized control offers optimal coordination, it is often __impractical__ for large-scale systems due to limitations in scalability, privacy, and robustness.

A natural alternative is distributed control, where each agent solves a local optimization problem using only information from its immediate neighbors. This distributed formulation retains cooperative behavior while reducing computational and communication burdens:
```math
\min_{u_i} \int_{0}^{T} \left(||x_i - x_i^{\text{ref}}||^2 + ||u_i||^2 + \alpha\sum_{j \in \mathcal{N}_i}||x_i - x_j||^2 \right) dt
```
Here, each agent independently computes its control input $u_i$ and exchanges limited information with neighbors, and collectively the network reaches a coordinated outcome.
"""

# ╔═╡ 44154c9f-8e5e-49a7-b6ec-f465e0769c88
md"""
## 2. Consensus Algorithms

So what exactly is __consensus__?
It refers to _reaching an agreement on a certain quantity of interest that depends on the states of all agents in a network._

A __consensus algorithm__ is a protocol that enables this agreement through _local communication_, by exchanging information only with neighboring agents.

### 2A. Applications of Consensus
In the earlier example of distributed control for autonomous vehicles, each agent must achieve consensus over shared variables such as _relative positions_ and _velocities_.
This coordination behavior is commonly known as the __flocking problem__.
"""

# ╔═╡ 40d12761-6ba5-4f92-aada-a26c9ddf5120
begin
	imgpath = joinpath(@__DIR__, "background_materials", "platooning.jpg")
	
	md"""
	$(PlutoUI.LocalResource(imgpath, :width => 300))
	
	*Figure: Vehicle platooning illustration*
	"""
end

# ╔═╡ 242f3a4c-ad0b-4619-80e6-1f1ee244b6f4
question_box(md"""This is just one example where consensus appears. Other examples include:
			 1. space rendezvous
			 2. power grid stability
			 3. federated learning
			 4. blockchain (distributed ledgers)
			 What shared variables require consensus in each of these settings?""")

# ╔═╡ a6c12abf-4303-45bc-99a7-7b55061013d6
Foldable(md"Answer...", md"""

		 1. space rendezvous - shared position
		 2. power grid stability - shared voltage
		 3. federated learning - shared global model
		 4. blockchain - shared transaction history
		 """)

# ╔═╡ cc94ceb2-e01a-4f10-9ee0-d0c906e5f46f
md"""
### 2B. Consensus Notation

We consider a network of decision-making agents with dynamics
$\dot{x}_i = u_i$,
where each agent communicates locally with its neighbors on a graph $G = (V, E)$, and with $|E| = N$ agents total.
The objective is to achieve asymptotic convergence to an agreement space such that
$x_1 = x_2 = \dots = x_n$.
Equivalently, this can be written as
$x = \alpha \mathbf{1}$,
where $\alpha \in \mathbb{R}$ represents the collective decision.

Let the adjacency matrix be $A = [a_{ij}]$.
The set of neighbors of agent $i$ is denoted by
$\mathcal{N}i = {, j \in V : a{ij} \neq 0 ,}$,
meaning that agent $i$ can communicate with agent $j$ whenever $a_{ij} \neq 0$.

In more general settings—such as mobile sensor networks or flocking—the graph can vary with time.
In this case, we write
$G(t) = (V, E(t))$, $A(t)$, and $\mathcal{N}_i(t)$
to denote the time-varying topology, adjacency matrix, and neighbor set, respectively.
"""

# ╔═╡ 453fb681-6f7b-42f2-9d10-0da7ad5b811c
md"""
### 2C. Distributed Consensus Algorithm
The distributed consensus algorithm takes the general form:
```math
\dot x_i(t) = \sum_{j \in \mathcal{N_i}} \left( x_j(t) - x_i(t) \right) + b_i(t)
```
where $b$ is an input bias, typically set to zero. A nonzero bias may represent, for instance, a desired inter-vehicle relative-position vector.

#### Average-Consensus
A common and instructive special case is the __average-consensus__ algorithm:
```math
\dot x_i(t) = \sum_{j \in \mathcal{N_i}} a_{ij} \left( x_j(t) - x_i(t) \right)
```
For graphs that are __connected__ and __undirected__ ($a_{ij} = a_{ji}$), the system satisfies a key invariance property: $\sum_i \dot x_i = 0$. This implies that the final consensus value must equal the average of the initial conditions:
```math
\alpha = \frac{1}{n}\sum_i x_i(0)
```
Hence, the algorithm asymptotically drives all agents to this average value for any initial state.
Average consensus finds wide application in sensor fusion, distributed estimation, and cooperative control problems within multi-agent systems.
"""

# ╔═╡ ee0fbf98-db44-4188-b623-3868a08c02b2
md"""
#### Graph Laplacian
The __Laplacian__ is a matrix defined as 
```math
L = [l_{ij}], \qquad
l_{ij} =
\begin{cases}
-1, & j \in \mathcal{N}_i, \\
|\mathcal{N}_i|, & j = i.
\end{cases}
```
The dynamics of the average-consensus algorithm can be compactly expressed as
```math
\dot x = -Lx
```
In an undirected graph, the matrix satisfies the sum-of-squares (SOS) property:
```math
x^\top Lx = \frac{1}{2}\sum_{(i,j)\in E}a_{ij}(x_j - x_i)^2
```
Defining the quadratic disagreement function $\varphi(x) = \frac{1}{2} x^T L x$, this yields the __gradient-descent__ algorithm:
```math
\dot{x} = -\nabla \varphi(x)
```
"""

# ╔═╡ 46a8121f-f1aa-4d22-8ef9-1d02f957767d
question_box(md"""What motivates the division by 2 in the quadratic disagreement function?""")

# ╔═╡ 85b74a70-a8df-4741-aa1b-d551d0e9bea2
Foldable(md"Answer...", md"It accounts for the double-counting of undirected edges in the graph.")

# ╔═╡ 524365a7-e799-4c55-acf4-88fcd5a716e2
md"""
### 2D. Demo: Connectivity and Consensus Performance
Spectral properties of $L$ affect the convergence of consensus algorithms. Try the demo below to see how __algebraic connectivity__ affects the performance in consensus algorithms.
"""

# ╔═╡ 3c5d6e70-6f43-11ef-3456-0123456789ab
md"""
##### Parameters

Adjust the parameters below to explore different graph topologies and consensus dynamics:

**Graph Structure:**

Number of nodes: $(@bind n_nodes Slider(5:50, default=15, show_value=true))

Topology: $(@bind topology Select(["random" => "Random (Erdős-Rényi)", "ring" => "Ring", "star" => "Star", "complete" => "Complete", "path" => "Path", "wheel" => "Wheel"]))

Connectivity (for random graphs): $(@bind connectivity Slider(0.1:0.05:0.9, default=0.3, show_value=true))

---

**Consensus Dynamics:**

Step size (α): $(@bind step_size Slider(0.01:0.01:0.5, default=0.1, show_value=true))

Number of iterations: $(@bind n_iterations Slider(10:10:200, default=100, show_value=true))
"""

# ╔═╡ 4d6e7f80-6f43-11ef-4567-0123456789ab
md"""
---
##### Graph Generation
"""

# ╔═╡ 5e7f8090-6f43-11ef-5678-0123456789ab
function create_graph(n::Int, topology_type::String, p::Float64=0.3)
	if topology_type == "complete"
		return complete_graph(n)
	elseif topology_type == "ring"
		return cycle_graph(n)
	elseif topology_type == "star"
		return star_graph(n)
	elseif topology_type == "grid"
		k = Int(ceil(sqrt(n)))
		# Create grid and take first n nodes
		g = grid([k, k])
		if nv(g) > n
			# Remove extra nodes
			for _ in 1:(nv(g) - n)
				rem_vertex!(g, nv(g))
			end
		end
		return g
	elseif topology_type == "path"
		return path_graph(n)
	elseif topology_type == "wheel"
		return wheel_graph(n)
	else  # random (Erdős-Rényi)
		return erdos_renyi(n, p)
	end
end

# ╔═╡ 6f809fa0-6f43-11ef-6789-0123456789ab
G = create_graph(n_nodes, topology, connectivity)

# ╔═╡ 7091a0b0-6f43-11ef-789a-0123456789ab
md"""
**Graph properties:**
- Nodes: $(nv(G))
- Edges: $(ne(G))
- Average degree: $(round(2 * ne(G) / nv(G), digits=2))
- Is connected: $(is_connected(G))
"""

# ╔═╡ 81a2b1c0-6f43-11ef-89ab-0123456789ab
md"""
---
##### Laplacian Matrix
"""

# ╔═╡ 92b3c2d0-6f43-11ef-9abc-0123456789ab
L = Matrix(laplacian_matrix(G))

# ╔═╡ a3c4d3e0-6f43-11ef-abcd-0123456789ab
md"""
Show Laplacian matrix: $(@bind show_laplacian CheckBox())
"""

# ╔═╡ b4d5e4f0-6f43-11ef-bcde-0123456789ab
if show_laplacian
	L
end

# ╔═╡ c5e6f500-6f43-11ef-cdef-0123456789ab
md"""
**Laplacian eigenvalues:**

The second smallest eigenvalue (algebraic connectivity) indicates how well-connected the graph is.
"""

# ╔═╡ d6f70610-6f43-11ef-def0-0123456789ab
begin
	λ = sort(eigvals(L))
	md"""
	- Smallest eigenvalue: $(round(λ[1], digits=6)) (should be ≈ 0)
	- Second smallest (algebraic connectivity): $(round(λ[2], digits=4))
	- Largest eigenvalue: $(round(λ[end], digits=4))
	"""
end

# ╔═╡ e7081720-6f43-11ef-ef01-0123456789ab
md"""
---
##### Initial Conditions
"""

# ╔═╡ f8192830-6f43-11ef-f012-0123456789ab
# Random initial values between 0 and 100
x₀ = rand(nv(G)) .* 100

# ╔═╡ 09293940-6f43-11ef-0123-0123456789ab
consensus_value = mean(x₀)

# ╔═╡ 1a3a4a50-6f43-11ef-1234-0123456789ab
md"""
**Initial statistics:**
- Mean (consensus value): $(round(consensus_value, digits=2))
- Std deviation: $(round(std(x₀), digits=2))
- Min: $(round(minimum(x₀), digits=2))
- Max: $(round(maximum(x₀), digits=2))
"""

# ╔═╡ 2b4b5b60-6f43-11ef-2345-0123456789ab
md"""
---
##### Consensus Dynamics
"""

# ╔═╡ 3c5c6c70-6f43-11ef-3456-0123456789ab
function consensus_evolution(L::Matrix, x₀::Vector, α::Float64, steps::Int)
	n = length(x₀)
	X = zeros(n, steps + 1)
	X[:, 1] = x₀
	
	for t in 1:steps
		# Discrete-time consensus update: x(t+1) = x(t) - α * L * x(t)
		X[:, t + 1] = X[:, t] - α * L * X[:, t]
	end
	
	return X
end

# ╔═╡ 4d6d7d80-6f43-11ef-4567-0123456789ab
X = consensus_evolution(L, x₀, step_size, n_iterations)

# ╔═╡ 6f8f9fa0-6f43-11ef-6789-0123456789ab
md"""
---
##### Visualization
"""

# ╔═╡ 92b2c2d0-6f43-11ef-9abc-0123456789ab
md"""
**Controls:**

Time step: $(@bind time_step Slider(0:n_iterations, default=0, show_value=true))

Node size: $(@bind node_size Slider(0.5:0.5:3.0, default=1.5, show_value=true))

Method: $(@bind method Select(["spring" => "Spring", "sfdp" => "SFDP", "stress" => "Stress"]))
"""

# ╔═╡ 5e7e8e90-6f43-11ef-5678-0123456789ab
md"""
**Statistics at time step $(time_step):**
- Mean: $(round(mean(X[:, time_step + 1]), digits=2))
- Variance: $(round(var(X[:, time_step + 1]), digits=4))
- Std deviation: $(round(std(X[:, time_step + 1]), digits=4))
- Distance to consensus: $(round(std(X[:, time_step + 1]), digits=4))
"""

# ╔═╡ a3c3d3e0-6f43-11ef-abcd-0123456789ab
begin
	# Get node colors based on current values
	node_values = X[:, time_step + 1]
	
	# Use the initial current min/max
	min_val_current = minimum(X[:, 1])
	max_val_current = maximum(X[:, 1])
	val_range = max_val_current - min_val_current
	
	# Create color palette from blue to red
	if val_range < 1e-6  # All values converged
		node_colors = [:purple for _ in 1:length(node_values)]
	else
		# Normalize to [0, 1]
		normalized_values = (node_values .- min_val_current) ./ val_range
		# Create colormap from blue (low) to red (high)
		node_colors = [RGB(v, 0.3*(1-v), 1-v) for v in normalized_values]
	end
	
	# Get layout based on method
	if method == "spring"
		layout_fn = spring
	elseif method == "sfdp"
		layout_fn = sfdp
	else
		layout_fn = stress
	end
	
	# Compute layout
	adj_mat = adjacency_matrix(G)
	pos = layout_fn(adj_mat)
	
	# Extract x and y coordinates
	xs = [p[1] for p in pos]
	ys = [p[2] for p in pos]
	
	# Create the plot
	p1 = plot(size=(700, 600), 
		      legend=false, 
		      framestyle=:none,
		      aspect_ratio=:equal,
		      title="Consensus State at t = $(time_step)\nConsensus: $(round(consensus_value, digits=2)) | Std: $(round(std(node_values), digits=3))")
	
	# Draw edges
	for e in edges(G)
		i, j = src(e), dst(e)
		plot!(p1, [xs[i], xs[j]], [ys[i], ys[j]], 
			  color=:gray, alpha=0.4, linewidth=1.5)
	end
	
	# Draw nodes
	scatter!(p1, xs, ys, 
		     markercolor=node_colors,
		     markersize=10*node_size,
		     markerstrokewidth=2,
		     markerstrokecolor=:black)
	
	p1
end

# ╔═╡ b4d4e4f0-6f43-11ef-bcde-0123456789ab
md"""
---
##### Node Value Evolution Over Time
"""

# ╔═╡ c5e5f500-6f43-11ef-cdef-0123456789ab
begin
	p2 = plot(0:n_iterations, X',
		alpha = 0.6,
		legend = false,
		xlabel = "Time step",
		ylabel = "Node value",
		title = "Consensus Convergence",
		linewidth = 2,
		size = (700, 400),
		margin = 5Plots.mm
	)
	
	# Add consensus line
	hline!(p2, [consensus_value],
		color = :red,
		linewidth = 3,
		linestyle = :dash,
		label = "Consensus value",
		legend = :topright
	)
	
	# Add current time marker
	vline!(p2, [time_step],
		color = :green,
		linewidth = 2,
		linestyle = :dot,
		alpha = 0.5
	)
end

# ╔═╡ d6f60610-6f43-11ef-def0-0123456789ab
md"""
---
##### Variance Over Time

"""

# ╔═╡ e7071720-6f43-11ef-ef01-0123456789ab
begin
	variances = [var(X[:, t]) for t in 1:(n_iterations+1)]
	
	p3 = plot(0:n_iterations, variances,
		xlabel = "Time step",
		ylabel = "Variance",
		title = "Convergence to Consensus",
		linewidth = 2,
		color = :purple,
		legend = false,
		size = (700, 400),
		margin = 5Plots.mm,
		yscale = :log10
	)
	
	vline!(p3, [time_step],
		color = :green,
		linewidth = 2,
		linestyle = :dot,
		alpha = 0.5
	)
end

# ╔═╡ f8182830-6f43-11ef-f012-0123456789ab
md"""
---
##### Analysis
A larger algebraic connectivity means faster convergence. This depends largely on the chosen topology. __Step size__ also affects stability.
- Too small: Slow convergence
- Too large: May cause oscillations or instability
- Optimal: Related to the largest eigenvalue of L (α < 2/λ_max for stability)

**Maximum safe step size for current graph:** α < $(round(2/maximum(λ), digits=3))
"""

# ╔═╡ 7e2909e5-0667-4589-b3df-24f48ae67fd8
md"""
## 3. ADMM for Consensus
The __Alternating Direction Method of Multipliers (ADMM)__, first introduced in the 1970s, is a powerful decomposition–coordination algorithm designed to solve large optimization problems by breaking them into smaller subproblems that can be solved in parallel and then coordinated toward a global solution.

To understand how ADMM applies to __consensus__, we first review several precursor methods that motivate its structure and intuition.

### 3A. Dual Ascent Method
We begin with the **convex optimization problem**:

```math
\begin{aligned}
&\text{minimize} && f(x) \quad \quad \quad \quad (1) \\
&\text{subject to} && Ax = b
\end{aligned}
```
The __Lagrangian__ is defined as:
```math
L(x,y) = f(x) + y^\top (Ax - b)
```
The __dual function__ is obtained by minimizing the Lagrangian over $x$:
```math
g(y) = \inf_{x} L(x,y) = -f^*(-A^\top y) - b^\top y
```
and the __dual problem__ seeks to maximize $g(y)$, where $f^*$ represents the convex conjugate of $f$.

The dual ascent iterations alternate between minimizing the Lagrangian over $x$ and updating the dual variable $y$:
```math
\begin{aligned}
x^{k+1} &:= \arg\min_x L(x, y^k) \\
y^{k+1} &:= y^k + \alpha^k (A x^{k+1} - b)
\end{aligned}
```
where __residual__ $(A x^{k+1} - b)$ acts as a gradient (or subgradient) direction.

However, this method comes with restrictive convergence assumptions: $f$ must be strictly convex, and $L$ must remain bounded in $x$ for every $y$. As a result, it is not widely applicable in practice.
"""

# ╔═╡ 8bcd8d66-fc1c-419f-8e36-374c2c22965a
md"""
### 3B. Dual Decomposition
The next precursor method is __dual decomposition__. Suppose now that the objective is separable: 
```math
f(x) = \sum_{i = 1}^N f_i(x_i)
```
and the matrix $A$ is partitioned so that $Ax = \sum_{i=1}^N A_ix_i$. This allows the Lagrangian to be expressed as:
```math
L(x, y) = \sum_{i=1}^{N} L_i(x_i, y)
= \sum_{i=1}^{N} \left( f_i(x_i) + y^{T} A_i x_i - \frac{1}{N} y^{T} b \right)
```

With this formulation, the $x$-minimization step decomposes into $N$ independent subproblems which can now be solved in parallel. This parallelization significantly reduces computation time and enhances scalability.
```math
\begin{aligned}
	x_i^{k+1} &:= \arg\min_{x_i} L_i(x_i, y^{k}) \\
	y^{k+1} &:= y^{k} + \alpha^{k} (A x^{k+1} - b)
\end{aligned}
```
"""

# ╔═╡ f75ec743-d448-432f-9511-b2c7a382873c
md"""
### 3C. Augmented Lagrangians and Method of Multipliers
The final precursor method is the __Augmented Lagrangian__ and __Method of Multipliers__.

The __Augmented Lagrangian__ introduces an additional quadratic penalty term:
```math
L_{\rho}(x, y) = f(x) + y^{T}(A x - b) + \frac{\rho}{2} \|A x - b\|_{2}^{2}
```
This corresponds to the optimization problem:
```math
\begin{aligned}
        &\text{minimize} \quad f(x) + \frac{\rho}{2}\|A x - b\|_{2}^{2} \\
        &\text{subject to} \quad A x = b
        \end{aligned}
```
This formulation is equivalent to Model (1), since the penalty term vanishes for any feasible $x$. 

Applying the dual ascent framework to this Lagrangian yields the algorithm known as the __Method of Multipliers__:
```math
\begin{aligned}
	x^{k+1} &:= \arg\min_{x} L_{\rho}(x, y^{k})\\
	y^{k+1} &:= y^{k} + \rho (A x^{k+1} - b)
\end{aligned}
```
This method enhances the robustness of the dual ascent algorithm, removing the need for strict convexity or boundedness assumptions on $f$. However, $f$ must still be convex and closed, and Slater's condition remains necessary to ensure strong duality.

__Note:__ the penalty parameter $\rho$ now serves as the algorithm’s step size. The quadratic penalty smooths the dual function but also scales its gradient, and the step size $\rho$ compensates for this effect.

Finally, even when $f$ is separable, the agumented Lagrangian $L_\rho$ is not, a key limitation that motivates the next method: __ADMM__
"""

# ╔═╡ b9146a5c-2898-4814-935a-6b1bfe7844c4
md"""
### 3D. ADMM: Alternating Direction Method of Multipliers
The ADMM algorithm blends the best of both worlds. It benefits from the separability and parallelization of dual ascent, while also carrying the superior convergence properties provided by the method of multipliers. 

Given the problem:
```math
\begin{aligned}
&\text{minimize} && f(x) + g(z) \\
&\text{subject to} && A x + B z = c
\end{aligned}
```
Its augmented Lagrangian is formed:
```math
L_{\rho}(x, z, y) = f(x) + g(z) + y^{T}(A x + B z - c) + \frac{\rho}{2}\|A x + B z - c\|_{2}^{2}
```
Finally, the algorithm iterates:
```math
\begin{aligned}
x^{k+1} &:= \arg\min_{x}\; L_{\rho}(x, z^{k}, y^{k}) \\
z^{k+1} &:= \arg\min_{z}\; L_{\rho}(x^{k+1}, z, y^{k}) \\
y^{k+1} &:= y^{k} + \rho\,\big(A x^{k+1} + B z^{k+1} - c\big)
\end{aligned}
```
Note that in the method of multipliers, the $x,z$-update occurs simultaneously. Here, they are updated in sequential or __alternating__ fashion, hence the name. 
"""

# ╔═╡ aec156de-2fce-47bd-b5ad-11e7c62e2a74
md"""
#### Assumptions for Convergence
__1.$f$ must be closed, proper, and convex__

This condition implies that the $x$-update is solvable.

__2. The un-augmented Lagrangian ($\rho=0$) has a saddle point $(x^*, z^*, y^*)$__

```math
L_0(x^*,z^*,y) \leq L_0(x^*,z^*,y^*) \leq L_0(x,z,y^*)
```
The first inequality implies that $y^*$ is dual optimal, and the second implies that $x^*,z^*$ is primal optimal. Together, they imply that strong duality holds.

With these two assumptions, the ADMM algorithm guarantees
* __residual convergence__: iterates approach feasibility
* __objective convergence:__ objective approaches optimal value
* __dual variable convergence__: iterates approach a dual optimal point

We refer the reader to the references for the proof of convergence. The important takeaway is that the augmented Lagrangian must be convex, so that the dual function is well-behaved and that the ADMM steps consistently decrease $L_\rho$.
"""

# ╔═╡ af74a473-4f23-45cc-883f-ed6ebe7167cf
question_box(md"""What happens when ADMM is used in the case when $f$ is not convex?""")

# ╔═╡ 3691eca5-604c-4947-8b44-6f0b16798187
Foldable(md"Answer...", md"Without convexity, the theoretical convergence guarantee is lost. Iterates could oscillate, diverge, or get trapped at local optima. However, the method can still be used heuristically.")

# ╔═╡ 6b93441f-e476-4c81-ad56-4df9222ade40
question_box(md"""What happens when ADMM is used in the case when the feasible region itself is nonconvex?""")

# ╔═╡ 58e61280-9add-40e1-b189-a4750591c804
Foldable(md"Answer...", md"ADMM loses its convexity-based guarantees: the subproblems may become ill-posed, and feasibility restoration through dual variables may fail. However, __in practice__, ADMM can still serve as a robust heuristic, often converging to feasible, locally optimal points when the problem is well-structured (e.g., low-rank constraints, sparsity, or separable nonlinearities).")

# ╔═╡ 01f2ceeb-9498-4c91-bd19-7f506752c005
md"""
#### Optimality Conditions
The necessary and sufficient optimality conditions for ADMM are primal feasibility:
```math
Ax^* + Bz^* - c = 0
```
and dual feasibility:
```math
\begin{aligned}
0 &\in \partial f(x^*) + A^\top y^* \\
0 &\in \partial g(z^*) + B^\top y^*
\end{aligned}
```
where $\partial$ denotes the subdifferential operator.

The stopping criteria for ADMM are typically based on __primal and dual residual tolerances__, which serve as practical bounds on the suboptimality of the current iterate.

#### Convergence in Practice
While ADMM can be relatively slow to achieve high-precision solutions, it typically converges to __moderate accuracy within a few tens of iterations__. In practice, this is often sufficient for large-scale problems where approximate solutions are acceptable or even preferable.

ADMM is particularly effective in __statistical and machine learning applications__, such as parameter estimation or regularized optimization, where extremely high accuracy offers diminishing returns. This contrasts with algorithms like __Newton’s method__ or __interior-point methods__, which are designed to achieve high precision efficiently but at significantly greater computational cost per iteration.

"""

# ╔═╡ 7fa5a486-0801-4977-8163-5a1fb02d59c9
md"""
### 3E. Consensus via ADMM
Consider the __global consensus problem__, where shared variables $x_i$ must reach a collective decision indicated by $z$:
```math
\begin{aligned}
& \text{minimize} \quad \sum_{i=1}^N f_i(x_i) \\
& \text{subject to} \quad x_i - z=0, \quad i=1,\dots,N
\end{aligned}
```
This yields the augmented Lagrangian:
```math
L_{\rho}(x_1, \ldots, x_N, z, y) = \sum_{i=1}^{N} \left( f_i(x_i) + y_i^{T}(x_i - z) + \frac{\rho}{2}\|x_i - z\|_{2}^{2} \right)
```
The ADMM then iterates:
```math
\begin{aligned}
x_i^{k+1} &:= \arg\min_{x_i} \left( f_i(x_i) + (y_i^{k})^{T}(x_i - z^{k}) + \frac{\rho}{2}\|x_i - z^{k}\|_{2}^{2} \right) \\
z^{k+1} &:= \frac{1}{N} \sum_{i=1}^{N} \left( x_i^{k+1} + \frac{1}{\rho} y_i^{k} \right) \\
y_i^{k+1} &:= y_i^{k} + \rho \left( x_i^{k+1} - z^{k+1} \right)
\end{aligned}
```
"""

# ╔═╡ 1de20b3e-5d93-4180-84d8-a2cbf2ab2966
md"""
#### ADMM in Expansion Planning: Progressive Hedging
A common application of ADMM for solving global consensus problems arises in the domain of __capacity expansion planning__ for power systems. This is typically formulated as a __two-stage optimization problem__, where the first stage determines investment decisions for generation, transmission, and/or storage, and the second stage evaluates system operation under various scenarios representing different operating conditions (e.g., demand levels, renewable outputs, or contingency events).

Let $s$ index the scenarios. For each scenario $s$, let $x_s$ denote the first-stage investment decisions and $w_s$ the corresponding second-stage operational decisions. The problem can be written as:
```math
\begin{aligned}
\min_{x_s,w_s,z} \quad & \sum_s p_s[c^\top x_s + f_s(w_s)] & \\
\text{s.t.} \quad & Tx_s + U_sw_s = d_s, \quad & \forall s \\
& x_s = z, \quad & \forall s
\end{aligned}
```
Here, $p_s$ denotes the probability (or weight) of scenario $s$. The __consensus constraint__ $x_s = z$, also called the __non-anticipativity constraint__, ensures that all scenario-specific investment decisions agree on a common global expansion plan $z$.

In this domain, the ADMM algorithm is given the name __Progressive Hedging__. 

First, the scenario-specific two-stage problems are solved, relaxing anticipativity:
```math
(x_s^{k+1}, w_s^{k+1}) = \arg\min_{x_s, w_s} \left[ c^\top x_s + f_s(w_s) + (y_s^{k})^\top (x_s - z^{k}) + \frac{\rho}{2}\|x_s - z^{k}\|^{2} \right]
```

Next, the consensus investment decision is updated. 
```math
z^{k+1} = \sum_{s} p_s \left( x_s^{k+1} + \frac{1}{\rho} y_s^{k} \right)
```

Finally, the multipliers are updated:
```math
y_s^{k+1} = y_s^{k} + \rho \left( x_s^{k+1} - z^{k+1} \right)
```
"""

# ╔═╡ c9484783-f791-4c68-a2e0-89f2a71fe851
md"""
### 3F. General Form Consensus
We now consider a __more general form of consensus__, beyond the global consensus setting. Suppose there are local variables $x_i \in \mathbb{R}^{n_i}$, and the objective function is separable across them:
$\sum_i f_i(x_i)$.

Each local variable $x_i$ contains a subset of components that correspond to certain components of a global variable $z$. We define a mapping $\mathcal{G}(i, j)$ from a local index $(i, j)$ to the corresponding global index $g$, such that
$(x_i)_j = z_{\mathcal{G}(i, j)}$.

As a motivating example, consider a __model fitting__ problem. Here, the global variable $z$ represents the full feature vector, while each processor (or node) $i$ holds a subset of the data. The corresponding local variable $x_i$ represents the subset of features in $z$ that appear in block $i$ of the data. This structure is typical in large-scale, high-dimensional datasets that are __sparse__ and __distributed__ across multiple computing nodes.

The general form consensus problem is:
```math
\begin{aligned}
& \text{minimize} \quad \sum_{i=1}^N f_i(x_i) \\
& \text{subject to} \quad x_i - \tilde z_{i}=0, \quad \forall i
\end{aligned}
```
where $(\tilde z_i)_j = z_{\mathcal{G}(i,j)}$.
"""

# ╔═╡ 373b57ee-68b2-4f2c-a94c-f06173c9ea2b
question_box(md"""As an exercise: write the augmented Lagrangian and the ADMM iterate updates for this problem. How does this differ from the algorithm for global consensus?""")

# ╔═╡ 4161b697-441e-4821-ba52-7f244886bd31
md"""
## 4. Application of ADMM to Distributed Control
Let's tie it all together! How can we conduct __distributed model predictive consensus__ via ADMM?

### 4A. Problem Setup

Consider the flocking problem in a network of double integrators. We assume dynamics in a discrete-time linear state:
```math
x_i(t+1) = A_ix_i(t) + B_iu_i(t), \quad \forall i
```

The objective is the infinite-horizon cost function:
```math
J = \sum_{t=0}^\infty \sum_{i=1}^N l_i(x_{\mathcal{N}_i}(t), u_{\mathcal{N}_i}(t))
```
where $x_{\mathcal{N}_i}(t), u_{\mathcal{N}_i}(t)$ are the concatenations of states and inputs of the neighbors of $i$.

The goal here is to find a __distributed__ optimal policy $\pi: \mathcal{X} \rightarrow \mathcal{U}$. Each agent $i$ should only depend on information from its neighbors in $G$. The finite-horizon optimization problem is:

```math
\begin{aligned}
\text{minimize} \quad 
& J = \sum_{t=0}^{T-1} \sum_{i=1}^{N} \ell_i(x_{\mathcal{N}_i}(t), u_{\mathcal{N}_i}(t))
+ \sum_{i=1}^{N} \ell_{if}(x_{\mathcal{N}_i}(T), u_{\mathcal{N}_i}(T)) \\
\text{subject to} \quad 
& x_i(t+1) = A_i x_i(t) + B_i u_i(t), \\
& x_{\mathcal{N}_i}(t) \in \mathcal{X}_i, \quad 
  u_{\mathcal{N}_i}(t) \in \mathcal{U}_i, \\
& i = 1, \ldots, N, \quad t = 0, 1, \ldots, T-1, \\
& x_{\mathcal{N}_i}(T) \in \mathcal{X}_{if}, \quad i = 1, \ldots, N.
\end{aligned}
```
"""

# ╔═╡ df6c20bd-fcb9-458f-a592-39f5a3781aa1
md"""
### 4B. Reformulation into General Form Consensus

We can formulate this into general form consensus. Let $\mathbf{x}_i$ be the local variable vector for agent $i$ that includes a copy of the state and input vectors of itself and all neighbors for the finite-horizon. Let $\mathbf{x}$ be the concatenation of all $\mathbf{x}_i$. The problem takes the form:
```math
\begin{aligned}
&\min_{\mathbf{x}_i \in \mathcal{X}_i} \quad \sum_{i=1}^{N} f_i(\mathbf{x}_i) \\
&\text{subject to} \quad \mathbf{x}_i - \bar{E}_i z = 0, \quad i = 1, \ldots, N
\end{aligned}
```
where $\bar{E}_i$ is a matrix that picks out components of $\mathbf{x}$ to match those of the local variable $\mathbf{x}_i$.

The Lagrangian takes the form:
```math
L_{\rho}(x, z, \lambda) 
= \sum_{i=1}^{N} \left[ 
f_i(x_i) 
+ \lambda_i^{T}(x_i - \bar{E}_i z) 
+ \frac{\rho}{2}\|x_i - \bar{E}_i z\|_2^2 
\right]
= \sum_{i=1}^{N} L_{\rho i}(x_i, z, \lambda)
```
and the ADMM iterates:
```math
\begin{aligned}
\mathbf{x}_i^{k+1} &= \arg\min_{\mathbf{x}_i \in \mathcal{X}_i} L_{\rho i}(\mathbf{x}_i, z^k, \lambda^k) \\
z^{k+1} &= \arg\min_{z} L_{\rho}(\mathbf{x}^{k+1}, z, \lambda^k) \\
\lambda_i^{k+1} &= \lambda_i^k + \rho \left( \mathbf{x}_i^{k+1} - \bar{E}_i z^{k+1} \right)
\end{aligned}
```
"""

# ╔═╡ e0141ff8-e681-45cc-b9a8-6908ed332fef
md"""
### 4C. Distributed MPC for Flocking Demo
A simple 2D multi-agent flocking problem solved using distributed model predictive control (MPC) with consensus ADMM.
"""

# ╔═╡ 326a7a64-73b0-4698-b82d-6ee5f4e9bbc9
begin
    # Problem parameters
    N_agents = 6          # Number of agents
    N_horizon = 10        # MPC horizon
    dt = 0.1             # Time step
    
    # ADMM parameters
    ρ = 10.0             # Penalty parameter
    max_iter = 20        # ADMM iterations
    
    # Cost weights
    Q_pos = 1.0          # Position tracking
    Q_vel = 0.5          # Velocity matching (flocking)
    R_control = 0.1      # Control effort
    
    # Communication radius
    comm_radius = 3.0
end

# ╔═╡ b37792db-7786-41b6-9014-58b24636e5e5
function get_neighbors(positions, comm_radius)
    """Find neighbors within communication radius"""
    N = size(positions, 2)
    neighbors = [Int[] for _ in 1:N]
    
    for i in 1:N
        for j in (i+1):N
            dist = norm(positions[:, i] - positions[:, j])
            if dist < comm_radius
                push!(neighbors[i], j)
                push!(neighbors[j], i)
            end
        end
    end
    return neighbors
end

# ╔═╡ 7c3aa41b-b0b6-42f7-bcea-a92a0f53e3f5
function agent_mpc_step(state_i, z_neighbors, λ_neighbors, neighbors, center, ρ, N_horizon, dt)
    """Solve local MPC for one agent using ADMM consensus"""
    # State: [x, y, vx, vy]
    # Control: [ax, ay]
    
    n_state = 4
    
    # Initialize trajectory
    x = zeros(n_state, N_horizon + 1)
    x[:, 1] = state_i
    
    # Compute desired velocity (average of neighbors + cohesion)
    target_vel = zeros(2)
    if !isempty(neighbors)
        for (j_idx, _) in enumerate(neighbors)
            target_vel += z_neighbors[j_idx][3:4, 1]
        end
        target_vel /= length(neighbors)
    else
        target_vel = state_i[3:4]
    end
    
    # Add cohesion component (move toward center)
    pos_to_center = center - state_i[1:2]
    cohesion_vel = 0.5 * pos_to_center
    target_vel = 0.7 * target_vel + 0.3 * cohesion_vel
    
    # Add consensus terms from dual variables
    if !isempty(neighbors)
        consensus_correction = zeros(2)
        for (j_idx, _) in enumerate(neighbors)
            consensus_correction -= λ_neighbors[j_idx][3:4, 1] / ρ
        end
        target_vel += consensus_correction / length(neighbors)
    end
    
    # Plan trajectory with smooth velocity transition
    for k in 1:N_horizon
        # Gradually transition to target velocity
        α_blend = min(1.0, k / 5.0)
        desired_vel = (1 - α_blend) * x[3:4, k] + α_blend * target_vel
        
        x[3:4, k+1] = desired_vel
        x[1:2, k+1] = x[1:2, k] + x[3:4, k+1] * dt
    end
    
    return x
end

# ╔═╡ 94207f26-34e1-418e-9e7b-e7fdfbb6ca8a
function consensus_admm_mpc(states, N_horizon, dt, ρ, max_iter)
    """Run consensus ADMM for distributed MPC"""
    N = size(states, 2)
    n_state = 4
    
    # Get communication graph
    neighbors = get_neighbors(states[1:2, :], comm_radius)
    
    # Compute flock center
    center = mean(states[1:2, :], dims=2)[:]
    
    # Initialize ADMM variables
    x_local = [zeros(n_state, N_horizon + 1) for _ in 1:N]
    z = [zeros(n_state, N_horizon + 1) for _ in 1:N]
    λ = [[zeros(n_state, N_horizon + 1) for _ in neighbors[i]] for i in 1:N]
    
    # Initialize with current state
    for i in 1:N
        x_local[i][:, 1] = states[:, i]
        z[i] = copy(x_local[i])
    end
    
    # ADMM iterations
    for iter in 1:max_iter
        # Update x (local MPC solutions)
        for i in 1:N
            z_neigh = [z[j] for j in neighbors[i]]
            x_local[i] = agent_mpc_step(states[:, i], z_neigh, λ[i], 
                                        neighbors[i], center, ρ, N_horizon, dt)
        end
        
        # Update z (consensus - average with neighbors)
        z_old = deepcopy(z)
        for i in 1:N
            if !isempty(neighbors[i])
                # Average velocity with neighbors for consensus
                z[i] = copy(x_local[i])
                for j in neighbors[i]
                    z[i][3:4, :] += x_local[j][3:4, :]
                end
                z[i][3:4, :] /= (1 + length(neighbors[i]))
            else
                z[i] = copy(x_local[i])
            end
        end
        
        # Update λ (dual variables)
        for i in 1:N
            for (j_idx, j) in enumerate(neighbors[i])
                λ[i][j_idx][3:4, :] += ρ * (x_local[i][3:4, :] - z[i][3:4, :])
            end
        end
    end
    
    # Apply velocity from consensus
    new_velocities = zeros(2, N)
    for i in 1:N
        new_velocities[:, i] = z[i][3:4, 2]  # Use next step velocity
    end
    
    return new_velocities
end

# ╔═╡ d475dc80-319b-409d-96d5-3706ac5c68c9
begin
    # Initialize agents in random positions
    Random.seed!(400)
    states = zeros(4, N_agents)  # [x, y, vx, vy]
    states[1:2, :] = randn(2, N_agents) * 3.0
    states[3:4, :] = randn(2, N_agents) * 0.5
    
    # Simulation
    T_sim = 100
    history = zeros(4, N_agents, T_sim)
    
    for t in 1:T_sim
        history[:, :, t] = states
        
        # Run distributed MPC with ADMM
        new_velocities = consensus_admm_mpc(states, N_horizon, dt, ρ, max_iter)
        
        # Update states
        states[3:4, :] = new_velocities  # Set consensus velocities
        states[1:2, :] += states[3:4, :] * dt  # Update positions
    end
end

# ╔═╡ 65b219d0-62fa-47cb-97d3-9f1285d956f5
begin
    # Visualization
    anim = @animate for t in 1:T_sim
        plot(size=(600, 600), xlim=(-8, 8), ylim=(-8, 8), 
             aspect_ratio=:equal, legend=false,
             title="Distributed MPC Flocking (ADMM)\nTime: $(round(t*dt, digits=1))s")
        
        # Plot trajectories
        for i in 1:N_agents
            plot!(history[1, i, max(1,t-20):t], history[2, i, max(1,t-20):t], 
                  alpha=0.3, color=i)
        end
        
        # Plot current positions and velocities
        for i in 1:N_agents
            scatter!([history[1, i, t]], [history[2, i, t]], 
                    markersize=10, color=i, markerstrokewidth=2)
            
            # Velocity arrows
            quiver!([history[1, i, t]], [history[2, i, t]], 
                   quiver=([history[3, i, t]], [history[4, i, t]]),
                   color=i, arrow=true, linewidth=2)
        end
        
        # Draw communication links
        neighbors = get_neighbors(history[1:2, :, t], comm_radius)
        for i in 1:N_agents
            for j in neighbors[i]
                if i < j
                    plot!([history[1, i, t], history[1, j, t]], 
                          [history[2, i, t], history[2, j, t]], 
                          color=:gray, alpha=0.2, linestyle=:dash)
                end
            end
        end
    end
    
    gif(anim, "flocking_admm.gif", fps=10)
end

# ╔═╡ a1dcc909-2116-4d53-825b-b5f686360bcf
md"""
### 4D. Benchmarking Distributed MPC against Central MPC
This section presents results from Summers et al. (2012), who apply __Distributed Model Predictive Consensus__ using the ADMM framework. The authors consider a flocking problem involving five agents, each with a six-dimensional state space representing position and velocity in 3D ($\mathbb{R}^6$). The objective penalizes both __neighbor disagreement__ and __control effort__. (Refer to the paper for detailed formulations of the system dynamics and noise modeling.)

Using a finite-horizon MPC with a horizon length of 10 and a 250-step closed-loop simulation, each local subproblem is solved in under 2 ms per ADMM iteration, demonstrating the __computational scalability__ of the distributed approach.

The first figure compares the agent trajectories from the __centralized solver__ (blue) and __ADMM__ (red), showing near-identical paths. The second figure shows the __percentage difference in total cost__ between ADMM and the centralized solver. Remarkably, ADMM achieves performance within __1.5%__ of the centralized solution after only __two iterations__.
"""

# ╔═╡ 1a351fa9-7dda-40a5-9bdd-788747344d94
begin
	imgpath2 = joinpath(@__DIR__, "background_materials", "admm_vs_central_trajectory.png")
	
	md"""
	$(PlutoUI.LocalResource(imgpath2, :width => 300))
	
	*Figure: Agent Trajectories Spatial Plot*
	"""
end

# ╔═╡ 8dc43ea8-6edb-4e8c-9f3a-bbdaf72860fa
begin
	imgpath3 = joinpath(@__DIR__, "background_materials", "admm_vs_central_cost.png")
	
	md"""
	$(PlutoUI.LocalResource(imgpath3, :width => 300))
	
	*Figure: Percent difference of ADMM cost with centralized solver*
	"""
end

# ╔═╡ eb7fc856-7182-43d3-9d0c-cb7b74260ef8
md"""
## 5. Connections with Other Lectures

### 5A. Dual Decomposition & SDDP
Recall the lecture on __SDDP (Stochastic Dual Dynamic Programming)__. This methodology is built upon __Dual Decomposition__ (Section 3B) principles, applied to multistage stochastic programs.

In this chapter, dual decomposition was used to decouple the problem __across agents__. In contrast, SDDP performs decomposition __across both scenarios and stages (time)__. The __forward propagation__ step, which simulates policies over sampled scenarios, is analogous to the __primal update__, while the __backward propagation__ step, which generates cuts to approximate the future cost-to-go function, is analogous to the __dual update__.

### 5B. ADMM & Kalman Filters
The previous lecture introduced the __Kalman Filter__, where each agent/sensor estimates its own system state from local measurements. However, __consensus was not enforced__: each local estimator opearted independently, and no mechanism ensured that neighboring agents' state estimates agreed. 

This lecture introduced __consensus optimization (via ADMM)__ that introduces a coordination layer: agents exchange information and iteratively enforce __agreement on shared variables__.

Conceptually, both frameworks involve __iterative information fusion__. However, one does not enforce consensus (only context sharing), while the other does.
"""

# ╔═╡ dd094928-b10a-4269-a5b0-90ec935254fb
md"""
## 6. Summary and Discussion
In this chapter, we introduced __consensus__:

* a mechanism for __agreement__ among agents through local communication
* convergence guaranteed under __connectivity__ and __convexity__
* enables coordination without centralized control (scalable, robust)

We also introduced __ADMM__, a powerful optimization framework:

* an Augmented Lagrangian method that combines __local optimization__ and __consensus enforcement__
* alternates between solving local subproblems and updating shared variables (dual updates)
* provides robust convergence and parallelizability for distributed optimization

Finally, we saw an application for __distributed MPC__:

* enables agents to cooperatively solve MPC problems without centralized control, enabling scalability, privacy, and robustness
"""

# ╔═╡ 58ae03d0-104b-4fc3-8983-b54548852c11
question_box(md"""Consensus and ADMM rely on cooperation and information exchange. What real-world systems or domains _fail_ to satisfy these assumptions (e.g., competitive markets, social media), and what modifications would be needed for such settings?""")

# ╔═╡ 1f24c378-3e4a-42d1-9a07-f3f226bb27b2
Foldable(md"Hint...", md"In such _adversarial_ or _strategic_ environments, the assumptions of standard consensus no longer hold. The optimization must be reformulated. It becomes __equilibrium-seeking__ rather than __agreement-seeking__.")

# ╔═╡ Cell order:
# ╟─462afff0-2ae3-4730-b48e-cf475fc9e14f
# ╟─195683a4-b093-46af-9fb2-0a8a67d996e8
# ╟─75bdf059-c8ac-4f9c-b023-3c010b4389cb
# ╟─fe6b8381-edeb-4d0c-87f5-4ecd2b7b9183
# ╟─44154c9f-8e5e-49a7-b6ec-f465e0769c88
# ╟─40d12761-6ba5-4f92-aada-a26c9ddf5120
# ╟─242f3a4c-ad0b-4619-80e6-1f1ee244b6f4
# ╟─a6c12abf-4303-45bc-99a7-7b55061013d6
# ╟─cc94ceb2-e01a-4f10-9ee0-d0c906e5f46f
# ╟─453fb681-6f7b-42f2-9d10-0da7ad5b811c
# ╟─ee0fbf98-db44-4188-b623-3868a08c02b2
# ╟─46a8121f-f1aa-4d22-8ef9-1d02f957767d
# ╟─85b74a70-a8df-4741-aa1b-d551d0e9bea2
# ╟─524365a7-e799-4c55-acf4-88fcd5a716e2
# ╟─3c5d6e70-6f43-11ef-3456-0123456789ab
# ╟─4d6e7f80-6f43-11ef-4567-0123456789ab
# ╟─5e7f8090-6f43-11ef-5678-0123456789ab
# ╟─6f809fa0-6f43-11ef-6789-0123456789ab
# ╟─7091a0b0-6f43-11ef-789a-0123456789ab
# ╟─81a2b1c0-6f43-11ef-89ab-0123456789ab
# ╟─92b3c2d0-6f43-11ef-9abc-0123456789ab
# ╟─a3c4d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d5e4f0-6f43-11ef-bcde-0123456789ab
# ╟─c5e6f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f70610-6f43-11ef-def0-0123456789ab
# ╟─e7081720-6f43-11ef-ef01-0123456789ab
# ╟─f8192830-6f43-11ef-f012-0123456789ab
# ╟─09293940-6f43-11ef-0123-0123456789ab
# ╟─1a3a4a50-6f43-11ef-1234-0123456789ab
# ╟─2b4b5b60-6f43-11ef-2345-0123456789ab
# ╟─3c5c6c70-6f43-11ef-3456-0123456789ab
# ╟─4d6d7d80-6f43-11ef-4567-0123456789ab
# ╟─5e7e8e90-6f43-11ef-5678-0123456789ab
# ╟─6f8f9fa0-6f43-11ef-6789-0123456789ab
# ╟─92b2c2d0-6f43-11ef-9abc-0123456789ab
# ╟─a3c3d3e0-6f43-11ef-abcd-0123456789ab
# ╟─b4d4e4f0-6f43-11ef-bcde-0123456789ab
# ╟─c5e5f500-6f43-11ef-cdef-0123456789ab
# ╟─d6f60610-6f43-11ef-def0-0123456789ab
# ╟─e7071720-6f43-11ef-ef01-0123456789ab
# ╟─f8182830-6f43-11ef-f012-0123456789ab
# ╟─7e2909e5-0667-4589-b3df-24f48ae67fd8
# ╟─8bcd8d66-fc1c-419f-8e36-374c2c22965a
# ╟─f75ec743-d448-432f-9511-b2c7a382873c
# ╟─b9146a5c-2898-4814-935a-6b1bfe7844c4
# ╟─aec156de-2fce-47bd-b5ad-11e7c62e2a74
# ╟─af74a473-4f23-45cc-883f-ed6ebe7167cf
# ╟─3691eca5-604c-4947-8b44-6f0b16798187
# ╟─6b93441f-e476-4c81-ad56-4df9222ade40
# ╟─58e61280-9add-40e1-b189-a4750591c804
# ╟─01f2ceeb-9498-4c91-bd19-7f506752c005
# ╟─7fa5a486-0801-4977-8163-5a1fb02d59c9
# ╟─1de20b3e-5d93-4180-84d8-a2cbf2ab2966
# ╟─c9484783-f791-4c68-a2e0-89f2a71fe851
# ╟─373b57ee-68b2-4f2c-a94c-f06173c9ea2b
# ╟─4161b697-441e-4821-ba52-7f244886bd31
# ╟─df6c20bd-fcb9-458f-a592-39f5a3781aa1
# ╟─e0141ff8-e681-45cc-b9a8-6908ed332fef
# ╟─326a7a64-73b0-4698-b82d-6ee5f4e9bbc9
# ╟─b37792db-7786-41b6-9014-58b24636e5e5
# ╟─7c3aa41b-b0b6-42f7-bcea-a92a0f53e3f5
# ╟─94207f26-34e1-418e-9e7b-e7fdfbb6ca8a
# ╟─d475dc80-319b-409d-96d5-3706ac5c68c9
# ╟─65b219d0-62fa-47cb-97d3-9f1285d956f5
# ╟─a1dcc909-2116-4d53-825b-b5f686360bcf
# ╟─1a351fa9-7dda-40a5-9bdd-788747344d94
# ╟─8dc43ea8-6edb-4e8c-9f3a-bbdaf72860fa
# ╟─eb7fc856-7182-43d3-9d0c-cb7b74260ef8
# ╟─dd094928-b10a-4269-a5b0-90ec935254fb
# ╟─58ae03d0-104b-4fc3-8983-b54548852c11
# ╟─1f24c378-3e4a-42d1-9a07-f3f226bb27b2
