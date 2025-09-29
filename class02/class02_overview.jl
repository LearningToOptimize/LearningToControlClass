### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ f656fd22-6dce-40da-b67a-febcee9bff08
using PlutoUI

# ╔═╡ a89a5e16-9c64-11f0-3c6f-772fd0a68437
md"""
## Sections
- [Root finding](class02_root_finding.jl)
- [Unconstrained minimization](class02_unconstrained_min.jl)
- [Constrained minimization](class02_constrained_min.jl)
- [Barrier vs. ALM](class02_methods_barrier_alm.jl)
- [Sequential Quadratic Programming](class02_sqp.jl)
"""


# ╔═╡ 46558377-74bc-4453-b2c2-273ccaf3d2d7
md"""
### How to read / run

1. Skim the **learning goals** below.
2. Open each subsection (links above). Read the short prose, then **run the cells top-to-bottom**.
3. Use the **sliders and toggles** to see how stepsizes, conditioning, and feasibility affect convergence.
4. Try at least **one “Check your understanding”** box per subsection.
"""

# ╔═╡ 61fc1519-10c9-4b57-ac8f-4b660c08b39d
md"""## Learning goals (what you’ll be able to do)

* **Pick and configure** an optimizer for small control problems (unconstrained & constrained).
* **Derive KKT conditions** and form the **QP/SQP subproblem** for a nonlinear program.
* **Explain differences** between **penalty**, **augmented Lagrangian (ALM)**, and **interior-point** methods, and when to prefer each.

**Why this matters.** It lets us map classic control tasks—LQR, MPC, trajectory optimization—into QPs/NLPs, exploit sparsity/warm starts, and choose a robust globalization strategy.
"""

# ╔═╡ 82c39021-20d1-4605-b10a-b1c6a29029e5
md"""## Big picture: why optimization for control?

* **Controller synthesis as optimization.** Many controllers are designed or run by solving optimization problems repeatedly.
* **MPC** solves a QP/NLP online each step—**warm-starts** and **sparsity** drive speed.
* **Trajectory optimization** uses nonlinear programming (often with collocation); robust **globalization** prevents solver stalls.
* **Learning-based control** wraps optimizers inside training loops (differentiable programming), so smoothness and convergence matter.

**Mental model (any time step (k))**
State (x_k) → **optimizer** solves a small problem → control (u_k) → plant evolves (x_{k+1}=f(x_k,u_k)).
The solver is part of the loop; reliability and speed translate directly to control performance.
"""

# ╔═╡ 515f280d-70e8-4b98-ba81-3861433594b3
md"""
## Notation I: derivatives & Jacobians

We keep **dimensions explicit** so shapes are clear.

* **Scalar-valued** (f:\mathbb{R}^n\to\mathbb{R})

  * Row-derivative (row gradient): (\dfrac{\partial f}{\partial x}\in\mathbb{R}^{1\times n})
  * Column gradient (our default): (\nabla f(x):=\left(\dfrac{\partial f}{\partial x}\right)^{!T}\in\mathbb{R}^{n})

  **First-order model:**
  [
  f(x+\Delta x)\approx f(x)+\nabla f(x)^{!T}\Delta x.
  ]

* **Vector-valued** (g:\mathbb{R}^m\to\mathbb{R}^n)

  * **Jacobian:** (\dfrac{\partial g}{\partial y}\in\mathbb{R}^{n\times m})

  **First-order model:**
  [
  g(y+\Delta y)\approx g(y)+\frac{\partial g}{\partial y},\Delta y.
  ]
"""

# ╔═╡ ad3d636c-571f-4671-a049-d7c4a9546058
md"""
## Notation II: gradient, Hessian & Taylor

* **Gradient (column):** (\nabla f(x)\in\mathbb{R}^{n}).
* **Hessian:** (\nabla^{2}f(x)\in\mathbb{R}^{n\times n}).
* **Shape check:** (\Delta x^{!T}\nabla^{2}f(x)\Delta x\in\mathbb{R}).

**Second-order Taylor expansion (near (x))**
[
f(x+\Delta x)\approx f(x)+\nabla f(x)^{!T}\Delta x+\tfrac{1}{2},\Delta x^{!T}\nabla^{2}f(x),\Delta x.
]

**Conventions in this chapter**

* We use the **column-gradient** convention in derivations.
* For constraints (g(x)=0) and (h(x)\le 0), Jacobians are stacked by rows so shapes align in KKT systems.
"""

# ╔═╡ 333765ac-6dc9-40fd-9b14-5905cd95cb30
md"""
KKT snapshot (for reference)

For
[
\min_x f(x)\quad \text{s.t.}\quad g(x)=0,\ \ h(x)\le 0,
]
the (informal) KKT conditions at a candidate (x^\star) are:

* **Stationarity:** (\nabla f(x^\star)+\nabla g(x^\star)^{!T}\lambda^\star+\nabla h(x^\star)^{!T}\mu^\star=0).
* **Primal feasibility:** (g(x^\star)=0,\ \ h(x^\star)\le 0).
* **Dual feasibility:** (\mu^\star\ge 0).
* **Complementarity:** (\mu_i^\star,h_i(x^\star)=0) for each inequality (i).

You’ll see how penalty/ALM/interior-point handle these conditions differently (explicitly, indirectly, or via barriers).
"""

# ╔═╡ a65e8de6-8f17-4adb-8359-cbe8a349eff8
md"""

"""

# ╔═╡ 12cbe2ab-9a12-4db6-a044-de0578bb49f3


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.71"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "984046fe41e3da5a07991eff195bd989ad3f9484"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

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

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

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
# ╠═f656fd22-6dce-40da-b67a-febcee9bff08
# ╠═a89a5e16-9c64-11f0-3c6f-772fd0a68437
# ╠═46558377-74bc-4453-b2c2-273ccaf3d2d7
# ╠═61fc1519-10c9-4b57-ac8f-4b660c08b39d
# ╠═82c39021-20d1-4605-b10a-b1c6a29029e5
# ╠═515f280d-70e8-4b98-ba81-3861433594b3
# ╠═ad3d636c-571f-4671-a049-d7c4a9546058
# ╠═333765ac-6dc9-40fd-9b14-5905cd95cb30
# ╠═a65e8de6-8f17-4adb-8359-cbe8a349eff8
# ╠═12cbe2ab-9a12-4db6-a044-de0578bb49f3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
