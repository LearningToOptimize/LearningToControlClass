### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 8f4c1994-9c6a-11f0-2d1d-296c30c0d0ed
md"""
# Unconstrained Minimization (Newton + Globalization)

*(Class 02 — interactive chapter section)*

[⬅ Back to Class 02 Overview](class02_overview.jl) · [⬅ Previous: Root-Finding](class02_root_finding.jl)

**In this section you will:**

* Review first/second-order optimality conditions.
* Derive **Newton’s method for minimization** from the quadratic model.
* Make Newton **robust** with **line search** (Armijo/Wolfe) and **trust regions**.
* See when/why to use **quasi-Newton (BFGS/L-BFGS)**.
* Practice on the **Rosenbrock** test problem (path & convergence diagnostics).

"""

# ╔═╡ db452cd5-1c97-40b5-9836-ac74ec96e950
md"""
## Big picture

We often recast tasks as
[
\min_{x\in\mathbb{R}^n} f(x),
]
then solve for a stationary point. Compared to generic root-finding (f(x)=0), minimization gives us **objective structure** (descent directions, merit functions, curvature) that we can exploit for **globalization** and **diagnostics**.

"""


# ╔═╡ fa119b30-970f-41a3-8ac6-2d132123d8c7
md"""

## First- and second-order conditions

Let (f:\mathbb{R}^n\to\mathbb{R}) be twice continuously differentiable.

* **First order (necessary):** at a local minimizer (x^\star),
  [
  \nabla f(x^\star)=0.
  ]
* **Second order:**

  * **Necessary:** (\nabla^2 f(x^\star)\succeq 0) (positive semidefinite).
  * **Sufficient (strict local min):** (\nabla^2 f(x^\star)\succ 0) (positive definite).

These are **local** conditions; nonconvex problems can have multiple minima/saddles.

"""

# ╔═╡ 44366de6-1152-45bb-b300-c766d64e85f8
md"""
## Quadratic Taylor model at (x)

The second-order Taylor expansion around (x) is
[
m_x(d) ;=; f(x) + \nabla f(x)^{!T}d + \tfrac12 d^{!T}\nabla^2 f(x),d.
]
Newton-type methods “approximately” minimize this model to choose the step (d).

* If (\nabla^2 f(x)\succ 0), (m_x) is strictly convex and has a unique minimizer.

"""

# ╔═╡ fe5dfe3c-f4d4-426f-a840-dc950c816222
md"""
## Derivation of the Newton step

Set (\nabla_d m_x(d)=0):
[
\nabla f(x) + \nabla^2 f(x),d ;=; 0
\quad\Longrightarrow\quad
d_{\text{N}} ;=; -\big(\nabla^2 f(x)\big)^{-1},\nabla f(x).
]
Update (x \leftarrow x + d_{\text{N}}) and repeat.

* **Quadratic convergence** near a nondegenerate minimizer ((\nabla^2 f(x^\star)) nonsingular).
* Each iteration requires solving a linear system with (\nabla^2 f(x)).

"""

# ╔═╡ 39bbcd3b-7ab7-45e3-ab27-eb94a25f507f
md"""
## Indefiniteness, overshooting, and saddles

* If (\nabla^2 f(x)) is **indefinite**, (d_{\text{N}}) may not be a **descent direction** ((\nabla f(x)^{!T} d < 0)).
* Even with (\nabla^2 f(x)\succ 0), a full step can **overshoot**; objective can increase.
* Newton may converge to a **saddle** if started in the wrong basin.

We fix this with **regularization** and **globalization**.

"""

# ╔═╡ 16b72035-3c8b-4c46-a1e7-35b062851422
md"""
## Positive-definite Hessian via regularization

If (\nabla^2 f(x)) is not PD, make it so:
[
H(x) \leftarrow \nabla^2 f(x) + \lambda I,\quad \lambda>0,
]
or use a **modified Cholesky** factorization to obtain (H(x)\succ 0). Then solve
[
H(x),d ;=; -\nabla f(x).
]
This ensures the **Newton direction is descent**: (\nabla f(x)^{!T} d < 0).

> In practice, (\lambda) is reduced toward (0) as we approach a minimizer.

"""

# ╔═╡ 50a7ea9f-72ad-4050-89c6-b3c7d02d07d8
md"""
Globalization I: line search

## Backtracking with Armijo / Wolfe

Given a descent direction (d), pick a step length (\alpha\in(0,1]).

* **Armijo (sufficient decrease):** find smallest (\alpha=c^j) (e.g., (c=\frac12)) such that
  [
  f(x+\alpha d) ;\le; f(x) + c_1,\alpha,\nabla f(x)^{!T} d, \quad 0<c_1<1.
  ]
* **(Strong) Wolfe:** also enforce curvature
  [
  \big|\nabla f(x+\alpha d)^{!T} d\big| ;\le; c_2,\big|\nabla f(x)^{!T} d\big|,\quad c_1 < c_2 < 1.
  ]

**Takeaway:** Armijo prevents overshooting; Wolfe also controls the slope, improving quasi-Newton updates.

"""

# ╔═╡ b4a652af-1fda-455c-b6d2-7e818120fdf2
md"""
Globalization II: trust regions

## Trust-region viewpoint

Instead of choosing (\alpha), **trust** the model only in a ball:
[
\min_{d}; m_x(d)
\quad \text{s.t.}\quad |d| \le \Delta.
]

* Solve approximately (e.g., **Cauchy point**, **dogleg**, truncated CG).
* If the actual decrease matches predicted decrease, **enlarge** (\Delta); else **shrink** and retry.

**Pros:** robust near indefiniteness and in poor scaling; integrates nicely with Hessian-free solvers.

"""

# ╔═╡ 49f460a5-e70e-472f-8149-f7dd6cf9e481
md"""
Quasi-Newton (BFGS & L-BFGS)

## When Hessians are pricey

Quasi-Newton builds a PD approximation (B_k \approx \nabla^2 f(x_k)) from gradients:

* **BFGS update** (ensures (B_{k+1}\succ 0) if (s_k^{!T}y_k>0))
  [
  s_k = x_{k+1}-x_k,\quad y_k=\nabla f(x_{k+1})-\nabla f(x_k),
  ]
  [
  B_{k+1} = B_k - \frac{B_k s_k s_k^{!T} B_k}{s_k^{!T} B_k s_k}
  + \frac{y_k y_k^{!T}}{y_k^{!T} s_k}.
  ]
* **L-BFGS** stores only a few ( (s_k, y_k) ) pairs → great for large-scale problems.

Use a **Wolfe line search** to satisfy the curvature condition (s_k^{!T}y_k>0).

"""

# ╔═╡ 1f90d0e6-8d3e-4aed-9cbb-748243e3a25a
md"""
Stopping criteria

## When to stop

Typical checks (use more than one):

* **Gradient norm:** (|\nabla f(x)| \le \varepsilon_g\cdot \max(1,|\nabla f(x_0)|)).
* **Step norm:** (|x_{k+1}-x_k| \le \varepsilon_x\cdot \max(1,|x_k|)).
* **Objective change:** (|f(x_{k+1})-f(x_k)| \le \varepsilon_f\cdot \max(1,|f(x_k)|)).
* **Iteration/time** limits.

Report **status** (converged, maxiter, failed-descent) and basic **history** for debugging.

"""

# ╔═╡ 5e586027-baa1-463b-a032-1b25f87fe270
md"""

Example: Rosenbrock function

## The Rosenbrock “banana”

[
f(x,y) = (1-x)^2 + 100,(y-x^2)^2.
]

* Narrow, curved valley; poor conditioning away from the minimizer (x^\star=(1,1)).
* Great to illustrate **line search**, **Hessian PD-repair**, and **path plots**.

**What to try (when you add code):**

* Start at ((-1.2,1.0)). Compare **pure Newton**, **regularized Newton**, **BFGS**.
* Plot contour lines and the **iterate path**. Track (|\nabla f|) on a **log scale**.


"""

# ╔═╡ ffe2800b-add1-4817-adeb-efc7660ead61
md"""
Conditioning & scaling

## Make life easier with scaling

Poor scaling slows convergence and breaks line searches.

* **Variable scaling:** change variables (x=S z) to make typical curvature similar.
* **Preconditioning:** solve (\tilde H d = -\tilde g) with (\tilde H = P^{-1/2} H P^{-1/2}).
* **Hessian-vector products:** Newton-CG/trust-region methods avoid forming (H) explicitly.
* **Automatic differentiation** gives accurate gradients/Hessians (or Hessian-vector products) cheaply.

"""

# ╔═╡ 0bb5977b-889d-408a-a073-c9a9b4e1eabc
md"""
Practical recipe (putting it together)

## A robust Newton-type solver (sketch)

1. Compute (g=\nabla f(x)). If (|g|) small → **stop**.
2. Compute/approximate (H=\nabla^2 f(x)) or (B\approx H).
3. **Make PD** (modified Cholesky or (+\lambda I)) to get a descent direction (d) from (H d=-g).
4. **Line search** (Armijo/Wolfe) or use a **trust region** to pick the step length/size.
5. Update (x\leftarrow x+\alpha d).
6. If quasi-Newton, update (B) via **BFGS**.
7. Repeat until stopping criteria met; record diagnostics.

"""

# ╔═╡ 2ef96814-04d3-466a-9d91-a43a3ae1f069
md"""
Check your understanding

1. **Newton direction is descent when (H\succ 0).** Show that if (H\succ 0) and (H d = -g\neq 0), then (g^{!T} d < 0).
2. **Rosenbrock Hessian.** Derive (\nabla f) and (\nabla^2 f) for Rosenbrock. Where is (H) poorly conditioned?
3. **Armijo parameter sensitivity.** Experiment (mentally or later in code) with (c_1\in[10^{-4},10^{-1}]) and backtracking factor (c\in{1/2,2/3}). How do step counts and robustness trade-off?
4. **BFGS curvature condition.** Explain why enforcing Wolfe guarantees (s^{!T}y>0) for smooth (f).

"""

# ╔═╡ 1dece802-261f-4f0e-a681-dbc6fcde5bbc
md"""
## Where to next

* Proceed to **Constrained Minimization & KKT** to extend Newton-type ideas to equality/inequality constraints and saddle-point systems.
* Keep the Rosenbrock notebook handy to test different globalization and quasi-Newton settings.

[⬅ Back to overview](class02_overview.jl)

[➡ Constrained minimization & KKT (next)](class02_constrained_min.jl) 
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
# ╠═8f4c1994-9c6a-11f0-2d1d-296c30c0d0ed
# ╠═db452cd5-1c97-40b5-9836-ac74ec96e950
# ╠═fa119b30-970f-41a3-8ac6-2d132123d8c7
# ╠═44366de6-1152-45bb-b300-c766d64e85f8
# ╠═fe5dfe3c-f4d4-426f-a840-dc950c816222
# ╠═39bbcd3b-7ab7-45e3-ab27-eb94a25f507f
# ╠═16b72035-3c8b-4c46-a1e7-35b062851422
# ╠═50a7ea9f-72ad-4050-89c6-b3c7d02d07d8
# ╠═b4a652af-1fda-455c-b6d2-7e818120fdf2
# ╠═49f460a5-e70e-472f-8149-f7dd6cf9e481
# ╠═1f90d0e6-8d3e-4aed-9cbb-748243e3a25a
# ╠═5e586027-baa1-463b-a032-1b25f87fe270
# ╠═ffe2800b-add1-4817-adeb-efc7660ead61
# ╠═0bb5977b-889d-408a-a073-c9a9b4e1eabc
# ╠═2ef96814-04d3-466a-9d91-a43a3ae1f069
# ╠═1dece802-261f-4f0e-a681-dbc6fcde5bbc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
