### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ cc19c250-9c64-11f0-069e-a1b411d0212b
md"[⬅ Back to Class 02 Overview](class02_overview.jl)"

# ╔═╡ c268beac-c0b0-4be1-b34e-bc5294751df9
md"# Root Finding and Fixed Points

In the first subsection of Chapter 2, we will be covering root finding and fixed points. We want to build an intuition for root-finding and fixed-point iteration. We also want to understand when/why simple fixed-point iteration converges. We will also derive Newton’s method from first principles. Apply it to an implicit Backward Euler step. Learn practical globalization (damping / line search on residual)."

# ╔═╡ 8a457e90-7526-4d5a-89d7-63588be4690d
md"""
## Root-Finding and Fixed Points (big picture)

* **Root-finding:** given $f:\mathbb{R}^n!\to!\mathbb{R}^n$, find ($x^\star$) with ($f(x^\star)=0$).
  Examples: steady states, solving nonlinear equations, implicit time-stepping.
* **Fixed point:** ($x^\star$) is a fixed point of (g) if ($g(x^\star)=x^\star$).
* **Bridge:** pick ($g(x)=x-\alpha f(x)$) with ($\alpha>0$). Then
  $$f(x^\star)=0 \quad\Longleftrightarrow\quad g(x^\star)=x^\star$$ .
* **Mindset:** start at ($x_0$) and iterate ($x_{k+1}=g(x_k)$) until nothing changes (within tolerance).
"""

# ╔═╡ 164e269d-21c7-4466-915f-95f5bcad08c9
md"""
Nuni - add a small julia visualization for 1d or 2d motion
"""

# ╔═╡ 30bb23ab-1ba1-4222-859d-bd095cc231f8
md"""## When does fixed-point iteration converge?
 
* Near $(x^\star)$, (g) behaves like its Jacobian ($J_g(x^\star)$) (linearization).
* **Contraction test:**

  * Scalar: ($|g'(x^\star)|<1$)
  * Vector: spectral radius ($\rho!\left(J_g(x^\star)\right)<1$)
* Smaller contraction ($\Rightarrow$) faster (linear) convergence.
* If the contraction fails ($(\ge 1)$), expect divergence or oscillations.
* Convergence is **local**: you need to start in the **basin of attraction**.
"""

# ╔═╡ 8336a88f-7a6b-473f-a3cf-711df97d2297
md"""
## Fixed-Point Iteration: minimal recipe

1. Choose ($g$) (often ($g(x)=x-\alpha f(x))$) and an initial guess ($x_0$).
2. Loop: ($x_{k+1}\leftarrow g(x_k)$).
3. Stop when either

   * **residual** ($|f(x_{k+1})|$) is small, or
   * **step size** ($|x_{k+1}-x_k|$) is small, or
   * **max iterations** hit.
4. **Report both** residual and step size to diagnose false convergence.

**Diagnostics tip.** Small step but large residual ($\Rightarrow$) stagnation (bad $(g)/(\alpha))$. Large step but small residual $(\Rightarrow)$ near solution but overshooting.
"""

# ╔═╡ 192e2d28-c502-4cc5-b365-21086d7e1bd4
md"""
## Tuning the simple iteration

* **Step size ($\alpha$):** too small $(\Rightarrow)$ slow; too large $(\Rightarrow)$ divergence/oscillation. Start modest, adjust cautiously.
* **Damping / relaxation:** $(x_{k+1}\gets (1-\beta)x_k+\beta,g(x_k))$, with $(0<\beta\le 1)$ stabilizes updates.
* **Rescaling/preconditioning:** choose $(g(x)=x-P,f(x))$ with a good matrix (P) (approx $(J_f(x^\star)^{-1})$) to speed convergence.
* **Better initial guesses** shrink the “distance to the basin”.

**Optimization link.** Gradient descent is fixed-point iteration on (\nabla F):
$(g(x)=x-\eta\nabla F(x))$ solves $(\nabla F(x^\star)=0)$.

"""

# ╔═╡ a8ab5084-6fb4-4f54-8a60-831c0a77e660
md"""
## Newton’s Method Derived (from linearization)

**TL;DR.** Instead of solving $(f(x)=0)$ directly, at the current $(x)$ solve a **linearized** system for a correction $(\Delta x)$.

* Linearize $(f)$ near (x):
  $$f(x+\Delta x);\approx; f(x) + J_f(x),\Delta x$$ .
  
* Set the approximation to zero and solve for ($\Delta x$):
  $$f(x) + J_f(x),\Delta x = 0
  \quad\Longrightarrow\quad
  \Delta x = -J_f(x)^{-1} f(x)$$
* Update and repeat: $(x \leftarrow x + \Delta x)$.

"""

# ╔═╡ 8658738a-b448-4aa9-9a1c-5a913cd3ceb1
md"""
## Newton: local behavior & requirements

* If (f) is smooth, (J_f(x^\star)) is **nonsingular**, and (x_0) is close enough to (x^\star), Newton converges **quadratically**:
  (|x_{k+1}-x^\star| \approx C |x_k-x^\star|^2).
* If (J_f(x^\star)) is singular or poorly conditioned, expect slow or erratic progress.
* Good **initialization** matters; Newton is a **local** method.

**Cost.** Each iteration solves a linear system with (J_f(x)): naive dense (O(n^3)), but real problems exploit **sparsity/structure** and reuse factorizations.
"""


# ╔═╡ 81832b4c-f010-41d9-8b62-3924bcb3669d
md"""## Globalization for Newton (residual line search)

Pure Newton can overshoot. A classic fix is a **line search on the residual norm**:

* Define merit (\phi(x)=\tfrac12 |f(x)|_2^2).
* Compute Newton step (\Delta x=-J_f(x)^{-1}f(x)).
* **Backtrack** on (\alpha\in(0,1]) until
  [
  \phi(x+\alpha\Delta x) ;\le; \phi(x) + c,\alpha,\nabla\phi(x)^{!T}\Delta x,
  \quad 0<c<1\ \text{(Armijo)}.
  ]
* Update (x\gets x+\alpha\Delta x).

**Alternatives.** Trust-region methods (e.g., Levenberg–Marquardt for least-squares) control step size by solving a **restricted** subproblem.
"""


# ╔═╡ 24cd83da-feae-4da0-9604-83767615c017
md"""
## Example: Backward Euler implicit step

Given dynamics (\dot{x}=f(x)) and step (h>0), **Backward Euler** seeks (x_{n+1}) via
[
x_{n+1} = x_n + h,f(x_{n+1}).
]
Define the residual in the unknown (x_{n+1}):
[
r(z) ;=; z - x_n - h,f(z) .
]
We solve the **root-finding** problem (r(z)=0) for (z=x_{n+1}).

* Newton step: solve (J_r(z),\Delta = -r(z)) with
  (J_r(z)=I - h,J_f(z)).
* **Fast local convergence** (often quadratic) once close.
* **Cost driver:** the linear solve; exploit **sparsity** in (J_f), reuse factorizations across timesteps, and **warm-start** with (z\approx x_n).

**Try it (when you add code).**

* Slider for (h) and initial guess (z_0).
* Compare pure Newton vs. residual line search.
* Plot (|r(z_k)|) per iteration (log scale).
"""

# ╔═╡ 060606a5-aca5-449e-a614-57b79b78a462
md"""
## Root-Finding via Minimization (and vice-versa)

* **Solve (f(x)=0)** by minimizing the least-squares objective
  (\min_x \tfrac12 |f(x)|_2^2).

  * **Gauss–Newton** uses (J_f(x)^{!T}J_f(x)\Delta = -J_f(x)^{!T} f(x)).
  * Good when (f) is a residual from data/physics (sum of squares).
* **Solve (\nabla F(x)=0)** (unconstrained minimization) with **Newton for gradients**:
  [
  \Delta x = -(\nabla^2 F(x))^{-1}\nabla F(x),\qquad x\gets x+\Delta x.
  ]
  This is “Newton for optimization” and appears in the next section.
"""

# ╔═╡ cb3d5657-0ef6-44bf-8818-95c48ad5595e
md"""
## Diagnostics & Pitfalls

* **Stagnation:** ($|x_{k+1}-x_k|\to 0$) but ($|f(x_k)|\not\to 0$).
  *Fix:* better (g)/preconditioner (P), damping, or switch to Newton.
* **Oscillation/divergence:** step too aggressive.
  *Fix:* relaxation ($\beta<1$), backtracking on residual, smaller ($\alpha$).
* **Singular/ill-conditioned Jacobian:** Newton step unstable.
  *Fix:* regularize (e.g., add ($\gamma I$)), trust-region, or switch to Gauss–Newton if least-squares.
* **Bad initial guess:** outside basin.
  *Fix:* problem-specific heuristics, continuation (homotopy), or multiple restarts.

"""

# ╔═╡ 44d93a6b-6575-438a-911f-6913d0ca02b5
md"""
## Ensuring descent / stability (regularization)

If the linear system for $(\Delta x)$ is unstable or poorly conditioned, **regularize**:
 $$\big(J_f(x) + \gamma I\big),\Delta = -f(x),\qquad \gamma>0$$
Then backtrack on ($\alpha$) and update ($x\gets x+\alpha\Delta$).

* Often called **damped Newton** (shrinks steps).
* Stabilizes solves and helps ensure decrease in the residual merit.
* Choose ($\gamma$) adaptively (smaller as the residual shrinks).
"""

# ╔═╡ a7af3229-b2ee-4dc1-991b-2196273a38db
md"""
## Where to go next

* Head to **Unconstrained Minimization (Newton + globalization)** to see how solving (\nabla F(x)=0) ties in and how to make Newton **reliably decrease** an objective.
* Later, we’ll re-use these ideas inside **KKT systems** for constrained problems.

[⬅ Back to overview](class02_overview.jl)

[➡ Unconstrained minimization (next)](class02_unconstrained_min.jl) 
"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
ForwardDiff = "~1.2.1"
PlutoUI = "~0.7.71"
PyPlot = "~2.11.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "90979212c71af3628dde6dc6c3039482aed6d33b"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "dc41303865a16274ecb8450c220021ce1e0cf05f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.2.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "d2c2b8627bbada1ba00af2951946fb8ce6012c05"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═cc19c250-9c64-11f0-069e-a1b411d0212b
# ╠═c268beac-c0b0-4be1-b34e-bc5294751df9
# ╠═8a457e90-7526-4d5a-89d7-63588be4690d
# ╠═164e269d-21c7-4466-915f-95f5bcad08c9
# ╠═30bb23ab-1ba1-4222-859d-bd095cc231f8
# ╠═8336a88f-7a6b-473f-a3cf-711df97d2297
# ╠═192e2d28-c502-4cc5-b365-21086d7e1bd4
# ╠═a8ab5084-6fb4-4f54-8a60-831c0a77e660
# ╠═8658738a-b448-4aa9-9a1c-5a913cd3ceb1
# ╠═81832b4c-f010-41d9-8b62-3924bcb3669d
# ╠═24cd83da-feae-4da0-9604-83767615c017
# ╠═060606a5-aca5-449e-a614-57b79b78a462
# ╠═cb3d5657-0ef6-44bf-8818-95c48ad5595e
# ╠═44d93a6b-6575-438a-911f-6913d0ca02b5
# ╠═a7af3229-b2ee-4dc1-991b-2196273a38db
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
