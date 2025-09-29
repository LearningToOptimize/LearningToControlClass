### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ cb245a30-f64e-4f2c-91f4-a50371b201a2
md"""
*(Class 02 — interactive chapter section)*

[⬅ Back to Class 02 Overview](class02_overview.jl) · [⬅ Previous: Methods (Penalty/ALM/IPM)](class02_methods_barrier_alm.jl)

**In this section you will:**

* Build the SQP **local model** (quadratic Lagrangian + linearized constraints).
* See the **QP subproblem** and how it relates to Newton on the KKT system.
* Learn **globalization** (merit / filter / trust-region) and **Hessian updates** (BFGS).
* Work through a small **toy example** and practical tips for control problems.

"""

# ╔═╡ 1adff7b2-d86b-4311-9626-22d52057f551
md"""
## Cell 2 — What is SQP?

## What is SQP?

**Idea.** Solve a nonlinear constrained problem by repeatedly solving a **quadratic program (QP)** built from local models.

* Linearize constraints; build a quadratic model of the **Lagrangian/objective**.
* Each iteration: solve a QP to get a step (d), then update (x \leftarrow x+\alpha d).
* With good Hessian information, SQP enjoys **strong local convergence** (often **superlinear**).


"""

# ╔═╡ 3d685462-e495-4c33-a2de-470e109bff78
md"""

## Cell 3 — Target problem & KKT recap

## Target NLP

[
\min_{x \in \mathbb{R}^n} \ f(x)
\quad\text{s.t.}\quad
g(x)=0,\quad h(x)\le 0,
]
with (g:\mathbb{R}^n!\to!\mathbb{R}^{m}) and (h:\mathbb{R}^n!\to!\mathbb{R}^{p}).

**KKT at a candidate optimum (x^\star):** there exist (\lambda\in\mathbb{R}^m), (\mu\in\mathbb{R}^p_{\ge 0}) with
[
\nabla f(x^\star)+ \nabla g(x^\star)^{!T}\lambda + \nabla h(x^\star)^{!T}\mu = 0, \quad
g(x^\star)=0, \quad
h(x^\star)\le 0, \quad
\mu\ge 0, \quad
\mu \odot h(x^\star)=0.
]

"""

# ╔═╡ 40e0acdf-f870-4ef6-9656-e909d5fb2e25
md"""

## Cell 4 — Local models at (x_k)

## From NLP to local models

At iterate (x_k) (with multipliers ((\lambda_k,\mu_k))):

**Quadratic model of the Lagrangian**
[
m_k(d) ;=; \nabla f(x_k)^{!T} d ;+; \tfrac{1}{2}, d^{!T} B_k, d,
]
with (B_k \approx \nabla^2_{xx}\mathcal{L}(x_k,\lambda_k,\mu_k)).

**Linearized constraints**
[
g(x_k)+\nabla g(x_k),d=0,
\qquad
h(x_k)+\nabla h(x_k),d \le 0.
]

"""

# ╔═╡ b23e9ac3-192d-41e6-9411-99214b612ab9
md"""

## Cell 5 — The SQP QP subproblem

## SQP subproblem (QP)

[
\begin{aligned}
\min_{d \in \mathbb{R}^n}\quad & \nabla f(x_k)^{!T} d + \tfrac12 d^{!T} B_k d\
\text{s.t.}\quad & \nabla g(x_k), d + g(x_k) = 0,\
& \nabla h(x_k), d + h(x_k) \le 0.
\end{aligned}
]

* Solve the QP (\Rightarrow) get step (d_k) and updated multipliers ((\lambda_{k+1},\mu_{k+1})).
* Update (x_{k+1} = x_k + \alpha_k d_k) (line search, **filter**, or **trust-region**).

"""

# ╔═╡ 64d7aa02-a814-4d0d-b365-90d70814932a
md"""

## Cell 6 — Algorithm sketch

## SQP: high-level algorithm

1. Initialize (x_0), multipliers ((\lambda_0,\mu_0)), and (B_0 \succ 0).
2. Form the QP at (x_k) using (B_k) and the linearized constraints.
3. Solve the QP (\Rightarrow) obtain (d_k), ((\lambda_{k+1},\mu_{k+1})).
4. **Globalize** (merit/filter/TR) to choose (\alpha_k) (and possibly restrict (d_k)).
5. Set (x_{k+1}=x_k+\alpha_k d_k); update (B_{k+1}) (e.g., **BFGS**).
6. Check convergence (stationarity, feasibility, complementarity) or continue.

"""

# ╔═╡ a725c268-9c6a-11f0-059f-fdcd96af48e3
md"""

## Cell 7 — Why this QP? (Newton–KKT connection)

## Newton–KKT connection

The QP KKT system is a **linearized** KKT system for the NLP.

* If (B_k) equals the exact (\nabla_{xx}^2\mathcal{L}(x_k,\lambda_k,\mu_k)), the SQP step is a **Newton step** (on the KKT equations).
* With **BFGS** (properly updated), SQP attains **superlinear** local convergence under standard assumptions (LICQ, strict complementarity, SOSC, Wolfe conditions on the merit).  
"""

# ╔═╡ ddb74901-0efe-4d4f-9c14-fc5098aaeced
md"""
## Cell 8 — Globalization I: (L_1) merit function

## Merit function line search

Define
[
\Phi_\mu(x)= f(x) + \mu\Big(,|g(x)|_1 + |h(x)^-|_1\Big),\quad
h^-_i(x)=\max(0,,h_i(x)).
]

* Choose (\mu) large enough to ensure **descent coupling** of feasibility and optimality.
* Backtrack on (\alpha\in(0,1]) to achieve an **Armijo** decrease in (\Phi_\mu).

**Caveat.** Large (\mu) can cause the **Maratos effect** (over-penalizing curvature, step rejection). See SOC below.

"""

# ╔═╡ af20e5bb-dc3e-4d13-af71-f44e23c4c544
md"""
## Cell 9 — Globalization II: filter (alternative to a big penalty)

## Filter method (accept if you improve *either* objective or feasibility)

Track pairs ((f,\theta)) with (\theta=|g(x)|_1+|h(x)^-|_1).

* Accept (x_{k+1}) if it **dominates** the filter: it reduces (f) for similar (\theta), or reduces (\theta) sufficiently.
* If the step doesn’t pass, try a **feasibility restoration** phase (focus on decreasing (\theta)).
  Filter avoids tuning a huge (\mu) and is widely used in practical SQP codes.

"""

# ╔═╡ 19da3f74-83da-4d6f-8bf0-55102050a58a
md"""
## Cell 10 — Trust-region SQP & step decomposition

## Trust-region SQP (TRSQP)

* Add (|d|\le \Delta) to the QP or use a **normal/tangential** split: first reduce constraint violation (normal step), then reduce (f) in the tangent space (tangential step).
* Adjust (\Delta) based on the ratio of **actual vs. predicted** improvement (as in unconstrained TR methods).
"""

# ╔═╡ 6b101d4e-7eff-4b91-8721-bd7bb7fafa17
md"""
## Cell 11 — Hessian models & updates

## Choosing (B_k): exact vs. quasi-Newton

* **Exact Lagrangian Hessian:** (B_k=\nabla^2_{xx}\mathcal{L}(x_k,\lambda_k,\mu_k)) (best local rate, expensive/less robust far away).
* **Gauss–Newton / curvature drop:** drop (\sum \mu_i \nabla^2 h_i) and (\sum \lambda_j \nabla^2 g_j) if residual-type; cheaper and often more stable.
* **BFGS / damped BFGS:** update (B_k\succ 0) with curvature condition (s_k^{!T}y_k>0) (enforce with Wolfe or damping).
* **Projected / null-space BFGS:** maintain positive definiteness on the **tangent space** to constraints.
"""

# ╔═╡ 7923cfd5-cd33-4001-9b71-ec8c17318ae8
md"""
## Cell 12 — The QP solver: structure matters

## Solving the QP efficiently

* **KKT structure:** sparse, block-saddle systems; use **Schur complements** or sparse symmetric indefinite factorizations.
* **Active-set QP solvers** (great with warm-starts; common in fast MPC).
* **Primal–dual IPM QP solvers** (robust for large QPs; excellent for dense inequality sets).
* **Warm-starting** ((x,\lambda,\mu)) and reusing factorizations across SQP iterations (and across time steps in MPC) is key.

"""

# ╔═╡ c1711266-ecab-4a93-af4b-7c1b0691a6a9
md"""
## Cell 13 — Toy example (local models)

## Toy example: local models

[
\min_{x\in\mathbb{R}^2}\ \tfrac12|x|^2
\quad\text{s.t.}\quad
g(x)=x_1^2+x_2-1=0,\quad
h(x)=x_2-0.2\le 0.
]

At (x_k):
[
\nabla f(x_k)=x_k,\quad B_k=I,\quad
\nabla g(x_k)=\begin{bmatrix}2x_{k,1} & 1\end{bmatrix},\quad
\nabla h(x_k)=\begin{bmatrix}0 & 1\end{bmatrix}.
]
Solve the QP for (d_k), then set (x_{k+1}=x_k+\alpha_k d_k) using a merit/filter/trust-region globalization.

*(When you add code: plot feasibility (|g|), violation (|h^-|), and objective vs. iteration.)*
"""

# ╔═╡ 03af9cf2-4ba7-4c31-9f36-88efe8e9eb37
md"""
## Where to next

* You now have the full pipeline: **KKT modeling → methods (Penalty/ALM/IPM) → SQP assembly**.
* In a control context (NMPC/trajectory optimization), plug in system sparsity and warm-starts to hit real-time targets.

[⬅ Back to overview](class02_overview.jl)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╠═cb245a30-f64e-4f2c-91f4-a50371b201a2
# ╠═1adff7b2-d86b-4311-9626-22d52057f551
# ╠═3d685462-e495-4c33-a2de-470e109bff78
# ╠═40e0acdf-f870-4ef6-9656-e909d5fb2e25
# ╠═b23e9ac3-192d-41e6-9411-99214b612ab9
# ╠═64d7aa02-a814-4d0d-b365-90d70814932a
# ╠═a725c268-9c6a-11f0-059f-fdcd96af48e3
# ╠═ddb74901-0efe-4d4f-9c14-fc5098aaeced
# ╠═af20e5bb-dc3e-4d13-af71-f44e23c4c544
# ╠═19da3f74-83da-4d6f-8bf0-55102050a58a
# ╠═6b101d4e-7eff-4b91-8721-bd7bb7fafa17
# ╠═7923cfd5-cd33-4001-9b71-ec8c17318ae8
# ╠═c1711266-ecab-4a93-af4b-7c1b0691a6a9
# ╠═03af9cf2-4ba7-4c31-9f36-88efe8e9eb37
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
