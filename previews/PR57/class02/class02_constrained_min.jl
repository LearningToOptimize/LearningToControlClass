### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ a509f9d8-9c6d-11f0-3db9-cb5fe2e85d64
md"""# Constrained Optimization (Equality & Inequality KKT)
 
[⬅ Back to Class 02 Overview](class02_overview.jl) 

[⬅ Previous: Unconstrained Minimization](class02_unconstrained_min.jl) 

[➡ Next: Methods (Penalty/ALM/IPM)](class02_methods_barrier_alm.jl)

**In this section you will:**

* Build the **geometry** of equality constraints and the **KKT** conditions.
* See the **Newton-on-KKT** linear system (saddle point) and when it’s well-posed.
* Contrast **full Newton** vs. **Gauss–Newton** on the KKT system.
* Extend to **inequality constraints** and understand **complementarity**.
 
"""

# ╔═╡ 57cd6d88-ea7e-4f5c-bc17-7fc65fb78e95
md"""## Equality-constrained minimization: geometry and conditions

**Problem:**
[
\min_{x\in\mathbb{R}^n} f(x) \quad \text{s.t.}\quad C(x)=0,\ \ C:\mathbb{R}^n\to\mathbb{R}^m.
]

**Geometric picture.** At an optimum on the manifold (C(x)=0), the negative gradient must lie in the tangent space:
[
\nabla f(x^\star)\ \perp\ \mathcal{T}_{x^\star}={p:\ J_C(x^\star)p=0}.
]
Equivalently, the gradient is a linear combination of the constraint normals:
[
\nabla f(x^\star)+J_C(x^\star)^{!T}\lambda^\star=0,\qquad C(x^\star)=0\quad(\lambda^\star\in\mathbb{R}^m).
]

**Lagrangian.** (L(x,\lambda)=f(x)+\lambda^{!T}C(x)).
"""

# ╔═╡ b7763163-fc68-4c56-bac9-c37de527858f
md"""
## Equality constraints: picture first

**Goal.** Minimize (f(x)) while staying on the surface (C(x)=0).

* **Feasible set as a surface.** Think of (C(x)=0) as a smooth surface embedded in (\mathbb{R}^n) (a manifold).
* **Move without breaking the constraint.** Tangent directions are the “along-the-surface” moves keeping (C(x)) unchanged to first order.
* **What must be true at the best point.** At (x^\star), there’s no downhill direction within the tangent space.
* **Normals enter the story.** If the gradient can’t point along the surface, it must be balanced by the normals ({J_C(x^\star)_{i:}^{!T}}), producing multipliers (\lambda^\star). 
"""

# ╔═╡ 8a08b045-3a1b-4601-a081-27a6a22d05e6
md"""
## From the picture to KKT (equality only)

For a regular local minimum:

1. **Feasibility:** (C(x^\star)=0).
2. **Stationarity:** (\nabla f(x^\star) + J_C(x^\star)^{!T}\lambda^\star = 0).

**Lagrangian viewpoint.** Define (L(x,\lambda)=f(x)+\lambda^{!T}C(x)). At a solution, (x^\star) is stationary for (L) w.r.t. (x), while (C(x^\star)=0) ensures feasibility.

**Interpreting (\lambda^\star).** Each (\lambda_i^\star) reflects how strongly the (i)-th constraint “pushes back”; it’s also a sensitivity of the optimal value to perturbations in (C_i).
"""

# ╔═╡ 907459d1-4a09-441c-a989-71ff687da873
md"""
## KKT system for equalities (first order) & Newton on KKT

**KKT (FOC):**
[
\nabla_x L(x,\lambda)=\nabla f(x)+J_C(x)^{!T}\lambda=0,\qquad C(x)=0.
]

**Newton on KKT (linearize both blocks):**
[
\begin{bmatrix}
\nabla^2 f(x) + \sum_{i=1}^{m}\lambda_i,\nabla^2 C_i(x) & ; J_C(x)^{!T}[2pt]
J_C(x) & ; 0
\end{bmatrix}
\begin{bmatrix}\Delta x\ \Delta\lambda\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(x)+J_C(x)^{!T}\lambda[2pt] C(x)
\end{bmatrix}.
]

**Notes.** This is a symmetric **saddle-point** system. Practical solves use block elimination (Schur complement) and sparse factorizations.
 """

# ╔═╡ aec59d16-c254-43b6-aee2-6261823fb7c3
md"""
## Newton on KKT: practice & safeguards

**Works best when:**

* (J_C(x^\star)) has **full row rank** (regularity).
* The **reduced Hessian** is **positive definite**.
* A **globalization** (e.g., merit/penalty line search) and mild **regularization** are present.

**Common safeguards:**

* **Regularize** the ((1,1)) block (e.g., (+\beta I)) to ensure a good search direction.
* **Merit/penalty line search** balancing feasibility vs. optimality.
* **Scaling** constraints to improve conditioning of the KKT system.
"""

# ╔═╡ a14a100c-5c92-42cc-899d-8c6eb0619368
md"""
## Gauss–Newton vs. full Newton (equality case)

* **Full Newton Lagrangian Hessian:**
  [
  \nabla_{xx}^2 L(x,\lambda)=\nabla^2 f(x)+\sum_{i=1}^m \lambda_i,\nabla^2 C_i(x).
  ]
* **Gauss–Newton approximation:** drop the constraint-curvature term:
  [
  H_{\text{GN}}(x)\approx \nabla^2 f(x).
  ]

**Trade-offs.**

* **Full Newton:** fewer iterations near the solution; costlier steps; less robust far away.
* **Gauss–Newton:** cheaper per step and often more stable; may need more iterations but competitive in wall-clock on many problems.
"""

# ╔═╡ 300c917e-e61c-4881-82d0-bc79ace66795
md"""
## Solving the KKT system: Schur complement (intuition)

Given
[
\begin{bmatrix} H & A^{!T}\ A & 0\end{bmatrix}
\begin{bmatrix}\Delta x\ \Delta\lambda\end{bmatrix}
=-
\begin{bmatrix} g\ c\end{bmatrix},
]
with (H\approx \nabla_{xx}^2 L), (A=J_C(x)), (g=\nabla f+J_C^{!T}\lambda), (c=C(x)).

* Eliminate (\Delta x): (\Delta x = -H^{-1}(g + A^{!T}\Delta\lambda)).
* Schur system in (\Delta\lambda):
  [
  (A H^{-1} A^{!T}),\Delta\lambda = c + A H^{-1} g.
  ]
* Then recover (\Delta x).
  Exploit **sparsity**: factor (H) once per iteration; reuse structure across iterations.

"""

# ╔═╡ 28a43bca-618a-4fec-a7eb-ad7a734ceac5
md"""
## Inequality-constrained minimization and KKT

**Problem:** (\min f(x)\ \text{s.t.}\ c(x)\ge 0,\ \ c:\mathbb{R}^n\to\mathbb{R}^p).

**KKT (FOC):**
[
\begin{aligned}
&\text{Stationarity:} && \nabla f(x)-J_c(x)^{!T}\lambda=0,\
&\text{Primal feasibility:} && c(x)\ge 0,\
&\text{Dual feasibility:} && \lambda\ge 0,\
&\text{Complementarity:} && \lambda^{!T}c(x)=0\quad(\lambda_i c_i(x)=0,\ \forall i).
\end{aligned}
]

**Interpretation.**

* **Active** constraints: (c_i(x)=0\Rightarrow \lambda_i) can be nonzero (acts like an equality).
* **Inactive** constraints: (c_i(x)>0\Rightarrow \lambda_i=0) (no influence on stationarity).
"""

# ╔═╡ 3b6300c0-d69f-4004-9b91-41116d0ce832
md"""
## Complementarity: intuition & Newton’s challenge

**What (\lambda_i c_i(x)=0) means.**

* Tight constraint ((c_i=0)) → can press back ((\lambda_i\ge 0)).
* Loose constraint ((c_i>0)) → no force ((\lambda_i=0)).

**Why naïve Newton struggles.**

* Complementarity brings **nonsmoothness** and **inequalities** ((\lambda\ge 0), (c(x)\ge 0)).
* Equality-style Newton can violate nonnegativity or bounce across the boundary.

**Two main strategies (preview).**

* **Active-set:** guess actives → solve equality-constrained subproblem → update the set.
* **Barrier / PDIP / ALM:** smooth or relax complementarity, use damped Newton, and drive the relaxation to zero.
 
"""

# ╔═╡ ab782dbb-5f3c-404f-87b7-091b4382b0aa
md"""
## Globalization with constraints: merit functions

To balance feasibility and optimality during updates ((x,\lambda)\to(x+\alpha\Delta x,\lambda+\alpha\Delta\lambda)), use a **merit/penalty** function, e.g.
[
\Phi_\mu(x) = f(x) + \mu,|C(x)|*1 \quad \text{(equality case)},
]
or for inequalities, a penalty on **violation** (v(x)=\sum_i \max(0,-c_i(x))).
Do a **backtracking line search** on (\Phi*\mu) to ensure robust progress.

*(You’ll see barrier and ALM variants in the next section.)*
 
"""

# ╔═╡ 0f722888-d66c-47ae-a1ee-1e2b7c9b4a58
md"""
## Conditioning & scaling

* **Scale constraints** so rows of (J_C) have comparable norms → better KKT conditioning.
* **Regularize** (H) when indefinite/ill-conditioned (modified Cholesky or (+\beta I)).
* **Exploit structure:** block-banded, sparse patterns common in trajectory problems.
* **Warm-starts** from previous solves (e.g., along continuation or time steps) improve robustness.
"""

# ╔═╡ f4d85409-9095-4bc0-b515-ae283e43f344
md"""
## Where to next

* Proceed to **Methods: Penalty vs. Augmented Lagrangian vs. Interior-Point** to see practical algorithms that *enforce* the KKT conditions reliably, including complementarity handling for inequalities.
* Later, we’ll assemble these pieces into **SQP**.

[➡ Methods (Penalty/ALM/IPM) (next)](class02_methods_barrier_alm.jl) · [⬅ Back to overview](class02_overview.jl)
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
# ╠═a509f9d8-9c6d-11f0-3db9-cb5fe2e85d64
# ╠═57cd6d88-ea7e-4f5c-bc17-7fc65fb78e95
# ╠═b7763163-fc68-4c56-bac9-c37de527858f
# ╠═8a08b045-3a1b-4601-a081-27a6a22d05e6
# ╠═907459d1-4a09-441c-a989-71ff687da873
# ╠═aec59d16-c254-43b6-aee2-6261823fb7c3
# ╠═a14a100c-5c92-42cc-899d-8c6eb0619368
# ╠═300c917e-e61c-4881-82d0-bc79ace66795
# ╠═28a43bca-618a-4fec-a7eb-ad7a734ceac5
# ╠═3b6300c0-d69f-4004-9b91-41116d0ce832
# ╠═ab782dbb-5f3c-404f-87b7-091b4382b0aa
# ╠═0f722888-d66c-47ae-a1ee-1e2b7c9b4a58
# ╠═f4d85409-9095-4bc0-b515-ae283e43f344
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
