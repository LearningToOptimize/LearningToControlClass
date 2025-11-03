### A Pluto.jl notebook ###
# v0.20.13

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

# ╔═╡ 7b896268-4336-47e2-a8b5-f985bfde51f5
begin
    import Pkg
 
    try
        Pkg.activate(@__DIR__)
    catch
        # Fallback if @__DIR__ not available in your setup
    end
    Pkg.instantiate()

    # Add any missing packages  
    for pkg in [
        "PlutoUI",
        "PlutoTeachingTools",
        "MarkdownLiteral",
        "ForwardDiff",
        "Plots"   
    ]
        if Base.find_package(pkg) === nothing
            Pkg.add(pkg)
        end
    end

    using Markdown, InteractiveUtils, PlutoUI, PlutoTeachingTools, MarkdownLiteral, LinearAlgebra, ForwardDiff
    import Plots                     
    const plt = Plots         
    plt.default(size=(700,420))      
end


# ╔═╡ 81ebc291-89f0-4c1e-ac34-d5715977dd86
md"""| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Arnaud Deza |
|  Date   | : | 29 of August, 2025 |"""

# ╔═╡ 9543f7bc-ab36-46ff-b471-9aa3db9739e4
ChooseDisplayMode()

# ╔═╡ 8969e78a-29b0-46d3-b6ba-59980208fe5b
md"""
#### Reference Material

[^cmu11]: Z. Manchester, Optimal Control (CMU 16-745) 2025 Lecture 3, Carnegie Mellon University, YouTube, 2025. [Online]. Available: [https://www.youtube.com/watch?v=f7yF0KOV-sI](https://www.youtube.com/watch?v=f7yF0KOV-sI)

[^cmu13]: Z. Manchester, Optimal Control (CMU 16-745) 2025 Lecture 4, Carnegie Mellon University, YouTube, 2025. [Online]. Available: [https://www.youtube.com/watch?v=lIuPIlDxLNU](https://www.youtube.com/watch?v=lIuPIlDxLNU)

[^cmu13]: Z. Manchester, Optimal Control (CMU 16-745) 2025 Lecture 5, Carnegie Mellon University, YouTube, 2025. [Online]. Available: [https://www.youtube.com/watch?v=bsBXk17rff4](https://www.youtube.com/watch?v=bsBXk17rff4)
"""

# ╔═╡ d90e9be0-7b68-4139-b185-6cbaad0d307e
md"""
### Some imports
"""

# ╔═╡ 342decc1-43fa-432a-9a9c-757a10ba6a5d
md"""
# Part 0: Overview of Lecture 2

Lecture 2 is the course’s optimization backbone: it makes explicit the idea that most control problems are optimization problems in disguise. 
We set the common language (gradients/Hessians, KKT systems, globalization) and the “solver toolbox” (penalty, augmented Lagrangian, primal–dual interior-point, SQP) that shows up everywhere else: 
* MPC is a QP/NLP solved online; 
* trajectory optimization is an NLP with sparse structure; 
* distributed control leans on operator splitting; 

and modern differentiable controllers backprop through these solvers. 

The point isn’t to prove every theorem—it’s to give you a reliable recipe for turning a control task into a well-posed QP/NLP and picking/configuring a solver that actually converges.

Positionally, this lecture is the hinge between “dynamics & modeling” (Class 1) and everything that follows: PMP/LQR reframed via KKT and Riccati; nonlinear trajectory optimization and collocation (Class 5); distributed MPC/ADMM (Class 8); GPU-accelerated solves (Class 9); and the learning side—adjoints for Neural DEs (Class 10) and optimization-as-a-layer for PINNs/neural operators (Classes 11–13). 

Concretely, you leave with (i) a map from control formulations to QP/NLP templates, (ii) rules of thumb for choosing between ALM, IPM, and SQP, and (iii) practical globalization (regularization + line search) so demos and projects are reproducible and fast.

### General structure of lecture

- **Root finding:** how implicit time-stepping (Backward Euler) turns into solving $r(x)=0$ each step.
- **Unconstrained minimization:** Newton’s method and why we need *globalization* (regularization + line search).
- **Constrained minimization:** KKT conditions in action for equality constraints.
- **Interior Point Method:** IPM, augmented lagragnian method.
- **SQP (Sequential Quadratic Programming):** how to reduce a nonlinear program to a sequence of easy QPs.
""" 

# ╔═╡ fd1ad74b-cb74-49a7-80b8-1a282abfdff2
md"""
# Part I — Unconstrained Minimization as Root Finding

In control, *implicit* simulators and optimizers show up everywhere:

- **Implicit time stepping** (e.g., Backward Euler) for stability on stiff dynamics.  
- **KKT systems** and **Newton steps** inside constrained optimization.  
- **Adjoints** in learning-to-control are solutions of linearized/adjoint equations.

All three reduce to solving nonlinear equations of the form $r(x)=0$. This section builds the minimal, reusable toolkit to do that reliably.

**Example we use:** one implicit Backward Euler step for $\dot x = f(x)$:
$x_{n+1} = x_n + h\,f(x_{n+1}) \;\Rightarrow\; r(x) \equiv x_n + h\,f(x) - x = 0.$

We compare two solvers for $r(x)=0$: a **fixed-point** iteration and **Newton’s method**, and relate what you see to the theory (contraction vs. quadratic convergence). 

## From differential equations to algebraic equations

Why not just use explicit Euler? For many control systems (pendulum near upright, contact, power systems) the dynamics can be **stiff**. Stability and larger steps favor **implicit** schemes, but those require solving $r(x)=0$ each step.

For Backward Euler (BE),
$r(x) = x_n + h\,f(x) - x,$
$\partial r(x) = h\,J_f(x) - I,$
where $J_f$ is the Jacobian of $f$. Good solvers exploit this structure: the residual is “identity minus something small” for small $h$, and the Jacobian carries the physics through $J_f$.

## Two algorithms we’ll compare

**Fixed-point (Picard)** defines $g(x):=x_n + h\,f(x)$ and iterates
$x_{k+1}=g(x_k).$
It converges if $g$ is a **contraction** near the root (intuitively, $h\,\|J_f\|$ small). The local linear rate is controlled by $\rho\!\left(J_g(x^\star)\right)=\rho\!\left(h\,J_f(x^\star)\right)$.

**Newton’s method** linearizes $r$ and solves for a correction:
$[\partial r(x_k)]\,\Delta x_k = -\,r(x_k), \quad x_{k+1}=x_k+\Delta x_k,$
so for BE,
$\underbrace{(h\,J_f(x_k)-I)}_{\text{BE Jacobian}}\Delta x_k = -\,(x_n+h\,f(x_k)-x_k).$

**Stopping & diagnostics.** We track the **residual norm** $\|r(x_k)\|$ and optionally the **step norm** $\|\Delta x_k\|$. Reporting both avoids “false convergence” when steps stagnate but residuals do not.

## What the theorems (briefly) say

- **Banach fixed-point theorem (Picard):** If $g$ is a contraction on a neighborhood of $x^\star$ (i.e., $\|J_g(x)\|\le q<1$), then $x_{k+1}=g(x_k)$ converges **linearly** to $x^\star$. For BE, $g'(x^\star)=h\,J_f(x^\star)$, so small $h$ or well-scaled $f$ helps.

- **Newton’s method:** If $r\in C^2$, $\partial r(x^\star)$ is nonsingular, and $x_0$ is close enough to $x^\star$, then the iterates are **quadratically** convergent:
$\|x_{k+1}-x^\star\| \le C\,\|x_k-x^\star\|^2.$
In practice we use **damping / line search** to reach the fast local regime from farther starts.

**Takeaway:** Picard is simple but fragile; Newton has higher per-iteration cost but far fewer iterations and better robustness for implicit steps.
"""

# ╔═╡ 49d5b2e6-eb29-478c-b817-8405d55170b1
begin
    """
    pendulum_dynamics(x)

    x = [θ; v], with θ̇ = v, v̇ = -(g/ℓ) sin(θ).
    Returns ẋ.
    """
    function pendulum_dynamics(x::AbstractVector)
        ℓ = 1.0
        g = 9.81
        θ, v = x[1], x[2]
        return @. [v, -(g/ℓ)*sin(θ)]
    end
end

# ╔═╡ 950c61b8-f076-4b9a-8970-e5c2841d75f2
md"""
## Residual and Jacobian

For the implicit step, define
$r(x) = x_n + h\,f(x) - x,$
$\partial r(x) = \dfrac{\partial}{\partial x}\big(x_n + h\,f(x) - x\big) = h\,J_f(x) - I.$
"""


# ╔═╡ 92841a2e-bc0d-40f8-8344-a5c398a67275
begin
    # Residual for BE step: r(x) = x_n + h f(x) - x
    residual(fun, x_n, h, x) = x_n .+ h .* fun(x) .- x

    function jac_residual(fun, x_n, h, x)
        if fun === pendulum_dynamics
            ℓ = 1.0; g = 9.81; θ = x[1]
            Jf = [ 0.0               1.0
                   -(g/ℓ)*cos(θ)     0.0 ]
            return Matrix(h * Jf - I)    # dense, safe for \
        else
            Jf = ForwardDiff.jacobian(fun, x)
            return Matrix(h * Jf - I)
        end
    end
end

# ╔═╡ 8813982c-8c9a-4706-91a8-ebadf9323a4f
md"""
## Root solvers

**Linear-algebra view.** Each Newton update solves
$(h\,J_f(x_k)-I)\,\Delta x_k = -\,r(x_k).$
For small $h$, the matrix is close to $-I$ (well-conditioned); as $h$ grows or the dynamics stiffen, conditioning deteriorates — that is when damping and scaling matter.



**Fixed point (Picard)** updates: $x_{k+1} = g(x_k) = x_n + h\,f(x_k)$.  
Converges locally if \(g\) is a contraction (roughly, small enough \(h\)).

**Newton** uses the true Jacobian \(hJ_f - I\); near a root it is typically quadratic (fewer iterations).

We’ll stop when $\|r(x_k)\|$ is small or a max iteration budget is reached.
"""

# ╔═╡ 4307a2f3-0378-4282-815b-9ed1fa532e1c
begin
    const SAFE = x -> max(x, eps())  # for log-scale plotting later

    """
    be_step_fixed_point(fun, x_n, h; tol=1e-8, maxiter=30)

    One implicit BE step via fixed-point. Returns (x_next, residual_norms).
    """
    function be_step_fixed_point(fun, x_n, h; tol=1e-8, maxiter=30)
        x = copy(x_n)               # natural initial guess
        errs = Float64[]
        for _ in 1:maxiter
            fx = fun(x)
            r  = x_n .+ h .* fx .- x
            push!(errs, norm(r))
            if errs[end] ≤ tol; break; end
            x = x_n .+ h .* fx      # Picard update using same fx
        end
        return x, errs
    end

    """
    be_step_newton(fun, x_n, h; tol=1e-10, maxiter=20)

    One implicit BE step via Newton on r(x)=0. Returns (x_next, residual_norms).
    """
    function be_step_newton(fun, x_n, h; tol=1e-10, maxiter=20)
        x = copy(x_n)
        errs = Float64[]
        for _ in 1:maxiter
            r  = residual(fun, x_n, h, x)
            push!(errs, norm(r))
            if errs[end] ≤ tol; break; end
            Dr = jac_residual(fun, x_n, h, x)
            Δx = - Dr \ r
            x .= x .+ Δx
        end
        return x, errs
    end
end

# ╔═╡ a45ed97f-f7c1-4ef5-9bc7-654e827f751b
md"""
### Controls

Initial state $x_n = [\theta,\, v]$:
- $\theta$ $(@bind θ0 Slider(-3.0:0.05:3.0, default=0.10, show_value=true))
- $v$ $(@bind v0 Slider(-6.0:0.1:6.0, default=0.00, show_value=true))

Step size $h$: $(@bind hstep Slider(0.1:0.1:0.2, default=0.1, show_value=true))  
Max iterations: $(@bind iters_max Select([10, 15, 20, 30]; default=20))

Compare: Fixed-point $(@bind show_fp CheckBox(default=true))
/ Newton $(@bind show_nt CheckBox(default=true))

$(@bind run Button("Run"))
"""



# ╔═╡ 5a17f83e-751b-4244-9c15-7165645bfe29
begin
    run                      

    x_n = [θ0, v0]

    x_fp, e_fp = show_fp ? be_step_fixed_point(pendulum_dynamics, x_n, hstep; maxiter=iters_max) : (x_n, Float64[])
    x_nt, e_nt = show_nt ? be_step_newton(pendulum_dynamics, x_n, hstep; maxiter=iters_max)     : (x_n, Float64[])

    efp = isempty(e_fp) ? Float64[] : max.(e_fp, eps())   # safe for log scale
    ent = isempty(e_nt) ? Float64[] : max.(e_nt, eps())

    p = plt.plot(title="Single-step convergence (‖r(x_k)‖)",
                 xlabel="iteration k", ylabel="‖r(x_k)‖", yscale=:log10)

    if !isempty(efp)
        plt.plot!(1:length(efp), efp, lw=2, marker=:circle, label="Fixed-point")
    end
    if !isempty(ent)
        plt.plot!(1:length(ent), ent, lw=2, marker=:utriangle, label="Newton")
    end
    p
end


# ╔═╡ 12c03202-58b2-482e-a65c-b83bc1f6eed1
md"""
**What to notice**

- For small enough \(h\), **fixed point** often converges, but slowly (roughly linear rate).
- **Newton** typically reaches the tolerance in a handful of steps (near-quadratic rate).
- Increase \(h\) to see fixed point struggle while Newton remains robust (until the Jacobian becomes ill-conditioned).
"""

# ╔═╡ 662d58d7-4c9c-4699-a846-cb6070c507d9
md"""
# Part II - Unconstrained Minimization — Newton, Regularization, and Line Search

**Learning goals**
- Apply Newton’s method to a nonconvex scalar objective.
- See why plain Newton can **increase** the objective (negative curvature).
- Stabilize Newton with **Hessian regularization**.
- Add **Armijo backtracking** to get robust progress.

We will use the toy function
$f(x) = x^4 + x^3 - x^2 - x$
which has both minima and maxima, so it’s perfect to show where Newton can go wrong.
"""

# ╔═╡ 27760607-b740-47dc-a810-c332baa2bd2d
begin
    #### Problem definition (scalar) ####
    f(x::Real)  = x^4 + x^3 - x^2 - x
    ∇f(x::Real) = 4x^3 + 3x^2 - 2x - 1
    ∇²f(x::Real)= 12x^2 + 6x - 2

    grad_f(x) = ∇f(x)
    hess_f(x) = ∇²f(x)

    #### Plot helpers ####
    function plot_fx(; xrange=(-1.75, 1.25), npts=1000, title="Objective")
        xs = range(xrange[1], xrange[2], length=npts)
        plt.plot(xs, f.(xs), lw=2, label="f(x)", xlabel="x", ylabel="f(x)", title=title)
    end

    function plot_trace(xhist; xrange=(-1.75, 1.25), npts=1000, title="")
        xs = range(xrange[1], xrange[2], length=npts)
        p = plt.plot(xs, f.(xs), lw=2, label="f(x)", xlabel="x", ylabel="f(x)", title=title)
        plt.scatter!(p, xhist, f.(xhist), marker=:x, ms=7, label="iterates")
        return p
    end
end


# ╔═╡ 7765e032-c520-4f97-b877-0d479f383f28
md"""
## 1) Inspect the objective

Below is the graph of \(f(x)\). The multiple critical points (max/min) make it a nice testbed for Newton.
"""

# ╔═╡ a098a49b-a368-4929-824c-385a06b88696
plot_fx(title="f(x) = x^4 + x^3 - x^2 - x")

# ╔═╡ a62f1b6a-87fe-4401-bc13-42166ca0e129
md"""
## 2) Plain Newton (no safeguards)

Newton update:


$x_{k+1} = x_k - \frac{\nabla f(x_k)}{\nabla^2 f(x_k)}$.


If $\nabla^2 f(x_k) < 0$, the step **ascends** along \(f\).

Use the sliders to choose a start and number of iterations.
"""

# ╔═╡ b2ddc048-4942-43bc-8faa-1921062d8c9c
begin
    # One Newton step (no regularization, no line search)
    newton_step_plain(x) = -grad_f(x) / hess_f(x)

    # Driver for plain Newton
    function newton_optimize_plain(x0; tol=1e-8, maxiters=10)
        x = float(x0)
        xhist = Float64[x]
        for _ in 1:maxiters
            g = grad_f(x)
            H = hess_f(x)
            if abs(g) <= tol || !isfinite(H)
                break
            end
            Δx = -g / H
            if !isfinite(Δx)
                break
            end
            x += Δx
            push!(xhist, x)
            if abs(grad_f(x)) <= tol
                break
            end
        end
        return xhist
    end
end


# ╔═╡ 6fd7a753-6db7-4c37-9e22-dc63dd3072c8
begin
    @bind x0_plain PlutoUI.Slider(-1.75:0.01:1.1, default=-1.50, show_value=true)
end


# ╔═╡ 49e0f5c3-fe14-42e1-9b3a-83c4447148a8
begin
    @bind iters_plain PlutoUI.Slider(1:20, default=8, show_value=true)
end

# ╔═╡ 096ebc95-f133-4ca3-b942-cf735faaa42b
begin
    xhist_plain_1 = newton_optimize_plain(x0_plain; maxiters=iters_plain)
    plot_trace(xhist_plain_1; title = "Plain Newton from x₀ = $(round(x0_plain,digits=3)) and iters = $(iters_plain)")
end


# ╔═╡ fdf41c76-4405-49e0-abfa-5c5193de99f4
md"""
## 3) Globalization: Regularization + Armijo backtracking

Two simple fixes:

1. **Regularization:** If $\nabla^2 f(x_k)\le 0$, replace it by $\nabla^2 f(x_k) + \beta $ with $\beta>0$ (LM-style), so the step is a **descent** direction.
2. **Armijo line search:** Scale the step by \(\alpha \in (0,1]\) until
$f(x_k+\alpha\Delta x) \le f(x_k) + b\,\alpha\,\nabla f(x_k)\,\Delta x$,
with typical $b=10^{-4}$, $c=0.5$ for backtracking.

We’ll expose both as toggles.
"""


# ╔═╡ d259c1b8-3716-4f80-b462-3b3daebb444d
md"""
### Controls

**Start x₀:** $(@bind x0_cmp Slider(-1.5:0.01:0.75, default=0.00, show_value=true))  
**Iterations:** $(@bind iters_cmp Slider(1:40, default=10, show_value=true))

**Regularization β₀:** $(@bind beta0 Slider(0.0:0.1:5.0, default=1.0, show_value=true))

**Armijo b:** $(@bind armijo_b Select([1e-4, 1e-3, 1e-2]; default=1e-4))  
**Armijo c (backtracking factor):** $(@bind armijo_c Slider(0.1:0.05:0.9, default=0.5, show_value=true))

**Show:**  
- Plain Newton $(@bind show_plain CheckBox(default=true))  
- Regularized $(@bind show_reg CheckBox(default=true))  
- Reg + Line Search $(@bind show_rls CheckBox(default=true))

**Layout:** $(@bind orientation Select(["row (1×N)", "column (N×1)"]; default="row (1×N)"))
"""


# ╔═╡ e012d7e0-0181-49d0-bb78-995378c4f87a
md"""
## 4) Takeaways

- **Plain Newton** is fast **if** you’re in a nice region and the Hessian is positive.
- **Regularization** turns the Newton step into a descent direction when curvature is negative or near-singular.
- **Armijo backtracking** fixes overshooting and makes progress predictable without hand-tuning step sizes.
- These building blocks generalize to higher-dimensional problems and constrained KKT systems (next section of the tutorial).

**Further experiments**
- Change the objective (keep the same drivers).
- Try different Armijo parameters $b, c$.
- Visualize **$|\nabla f(x_k)|$** vs iteration to see convergence.
"""


# ╔═╡ 0a3eb748-fab3-4f8b-993e-2246c32fb6aa


# ╔═╡ 3989b8e1-ac1f-430d-9d72-e298ba7ae0ca


# ╔═╡ e466ffb2-fcc8-4c3b-9059-7fd68a4265f2


# ╔═╡ 09a301c7-c1f4-4890-afe9-fbc1d7c3b905


# ╔═╡ 53020b8a-4948-467a-8629-bef9496d0374


# ╔═╡ 26c887ce-d95b-4e38-9717-e119de0e80ca
md"""
# Part III - Constrained Optimization (KKT) 

#### Setup (equality constraints)
We solve
```math
\min_{x\in\mathbb{R}^n} f(x) \quad \text{s.t.}\ C(x)=0,\qquad C:\mathbb{R}^n\to\mathbb{R}^m.
```

**Geometric picture.** At an optimum on the manifold `C(x)=0`, the gradient is orthogonal to the tangent space:

```math
\nabla f(x^\star)\ \perp\ \mathcal{T}_{x^\star}=\{p:\ J_C(x^\star)p=0\}.
```

Equivalently, the gradient is a combination of the constraint normals:

```math
\nabla f(x^\star)+J_C(x^\star)^{\!T}\lambda^\star=0,\qquad C(x^\star)=0.
```

**Lagrangian.** Define $L(x,\lambda)=f(x)+\lambda^{\!T}C(x)$.
"""

# ╔═╡ a1b7f614-710a-4ce8-815b-2f94754088c4
md"""
#### From conditions to a solver (Newton/SQP on the KKT system)

Linearizing feasibility and stationarity gives the **saddle-point** (KKT) system:

```math
\begin{bmatrix}
H & J_C(x)^{\!T} \\
J_C(x) & 0
\end{bmatrix}
\begin{bmatrix}\Delta x\\ \Delta\lambda\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(x)+J_C(x)^{\!T}\lambda\\
C(x)
\end{bmatrix},
\qquad
H \approx \nabla^2 f(x) + \sum_{i=1}^m \lambda_i\,\nabla^2 C_i(x).
```

Two common choices for `H`:

* **Full Newton (Hessian of the Lagrangian):** as written above (fast near a solution).
* **Gauss–Newton/SQP:** drop the constraint-curvature term so $H\approx\nabla^2 f(x)$ (often more robust far from the solution).

> Practical: this system is symmetric indefinite; block elimination (Schur complement) and sparse factorizations are standard.
"""

# ╔═╡ 6fd82e73-6fef-4f7f-b291-f94cbac0d268
md"""
#### Inequality constraints (KKT, first-order)

For $c(x)\ge 0$,

```math
\begin{aligned}
&\text{Stationarity:} && \nabla f(x)-J_c(x)^{\!T}\lambda=0,\\
&\text{Primal feasibility:} && c(x)\ge 0,\\
&\text{Dual feasibility:} && \lambda\ge 0,\\
&\text{Complementarity:} && \lambda^{\!T}c(x)=0\quad(\lambda_i\,c_i(x)=0\ \forall i).
\end{aligned}
```

**Interpretation.** Active constraints ($c_i(x)=0$) may carry nonzero multipliers; inactive ones ($c_i(x)>0$) have $\lambda_i=0$.
"""

# ╔═╡ 52718e0b-f958-445e-b9b6-9e5baf09e81a


# ╔═╡ edce5b27-9af8-4010-9d9f-60681b2f427c


# ╔═╡ c139ccc9-0355-4c41-8d82-0c97fef50900


# ╔═╡ 43c5ccb9-d51e-4d73-b7be-f8b9dd599130
md"""
# Part III Code - Equality Constrained Minimization (KKT) 

We solve

```math
\min_x\; f(x) \quad \text{s.t.}\; c(x)=0
```

with one equality constraint. The **KKT system** for a Newton/SQP step is

```math
\begin{bmatrix} H & J^\top \\ J & 0 \end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta \lambda \end{bmatrix}
= - \begin{bmatrix} \nabla f(x)+J^\top \lambda \\ c(x) \end{bmatrix},
\quad J=\nabla c(x)^\top,\; H\approx\nabla^2\!L(x,\lambda).
```

We'll compare:

* **Gauss–Newton/SQP**: $H=\nabla^2 f$ (drop constraint curvature) ⇒ robust PD top-left.
* **Full**: $H=\nabla^2 f + \lambda \nabla^2 c$.

We’ll also use a simple merit function $\phi(x)=f(x)+\frac{\rho}{2}\,c(x)^2$ with Armijo backtracking.
"""

# ╔═╡ e0a7f431-61dd-4df2-b53c-d81c7a302baf
begin
    # === Objective & constraint (KKT section) ===
    Q_kkt     = Diagonal([0.5, 1.0])
    xstar_kkt = [1.0, 0.0]

    # Scalar objective (use dot() to ensure a Float64, not 1×1)
    f_kkt(x::AbstractVector)      = 0.5 * dot(x .- xstar_kkt, Q_kkt * (x .- xstar_kkt))
    ∇f_kkt(x::AbstractVector)     = Q_kkt * (x .- xstar_kkt)
    ∇²f_kkt(::AbstractVector)     = Q_kkt  # constant SPD 2×2

    # One equality constraint: c(x) = x1^2 + 2x1 - x2
    c_kkt(x::AbstractVector)      = x[1]^2 + 2x[1] - x[2]
    ∇c_kkt(x::AbstractVector)     = [2x[1] + 2.0, -1.0]                # length-2
    ∇²c_kkt(::AbstractVector)     = [2.0 0.0; 0.0 0.0]                 # constant 2×2
end



# ╔═╡ 3e3d6e18-f922-4e91-8eaf-bc26b8f91757
begin
    function landscape_plot_kkt(; xlim=(-4,4), ylim=(-4,4), nsamp=120, show_colorbar=false)
        xs = range(xlim[1], xlim[2], length=nsamp)
        ys = range(ylim[1], ylim[2], length=nsamp)
        Z  = [ f_kkt([x,y]) for x in xs, y in ys ]

        p = plt.contour(xs, ys, Z;
                        levels=18,
                        colorbar=show_colorbar,
                        xlabel="x₁", ylabel="x₂",
                        title="Objective contours & constraint (c(x)=0)",
                        legend=:topleft,
                        aspect_ratio=:equal,
                        xlims=(xlim[1], xlim[2]),
                        ylims=(ylim[1], ylim[2]))

        xc = collect(xs)
        yc = @. xc^2 + 2xc
        plt.plot!(p, xc, yc; lw=2, label="c(x)=0")
        plt.scatter!(p, [xstar_kkt[1]], [xstar_kkt[2]]; marker=:star5, ms=8, label="x⋆")

        plt.xlims!(p, (xlim[1], xlim[2]))
        plt.ylims!(p, (ylim[1], ylim[2]))
        return p
    end

    plot_path_kkt!(p, X) = begin
        plt.plot!(p, X[1,:], X[2,:]; marker=:x, ms=6, lw=1.5, label="iterates")
        plt.xlims!(p, (-4,4)); plt.ylims!(p, (-4,4))
        p
    end
end



# ╔═╡ 31407c3f-8f3d-4dcc-bc49-cedd8ca14013
begin
    # === KKT step + simple merit line search (KKT section) ===

    ϕ_kkt(x; ρ=1.0)  = f_kkt(x) + 0.5*ρ*c_kkt(x)^2
    ∇ϕ_kkt(x; ρ=1.0) = ∇f_kkt(x) .+ ρ * c_kkt(x) .* ∇c_kkt(x)

    """
    kkt_step_kkt(x, λ; method=:gn, δ=1e-8)

    Solve the 3×3 KKT system:
      [ H  C';  C  0 ] [Δx; Δλ] = -[ ∇f + C'λ;  c ]
    with C = ∇c(x) as a 1×2 row, H = ∇²f (Gauss–Newton) or ∇²f + λ∇²c (Full).
    A small ridge δ is added to H on the diagonal for numerical robustness.
    """
    function kkt_step_kkt(x::AbstractVector, λ::Real; method::Symbol=:gn, δ::Float64=1e-8)
        g   = ∇f_kkt(x)                 # 2
        cv  = ∇c_kkt(x)                 # length-2
        C   = reshape(cv, 1, :)         # 1×2
        H   = Matrix(∇²f_kkt(x))
        if method === :full
            H .+= λ .* ∇²c_kkt(x)
        end
        # add small ridge on diagonal (avoid UniformScaling arithmetic)
        @inbounds for i in 1:size(H,1); H[i,i] += δ; end

        K   = [H  C';  C  0.0]          # 3×3
        rhs = -vcat(g .+ λ .* cv, c_kkt(x))
        Δ   = try
            K \ rhs
        catch
            pinv(K) * rhs
        end
        Δx  = Δ[1:2];  Δλ = Δ[3]
        return Δx, Δλ
    end

    """
    kkt_solve_kkt(x0, λ0; iters=8, method=:gn, linesearch=true, ρ=1.0, b=1e-4, cdec=0.5)

    A few KKT iterations with optional Armijo backtracking on ϕ_kkt.
    Returns (X, Λ) with X ∈ ℝ^{2×(iters+1)}.
    """
    function kkt_solve_kkt(x0, λ0; iters=8, method=:gn, linesearch=true, ρ=1.0, b=1e-4, cdec=0.5)
        x = copy(x0); λ = float(λ0)
        X = reshape(x, 2, 1); Λ = [λ]
        for _ in 1:iters
            Δx, Δλ = kkt_step_kkt(x, λ; method=method)
            α = 1.0
            if linesearch
                φx = ϕ_kkt(x; ρ=ρ)
                gφ = ∇ϕ_kkt(x; ρ=ρ)
                slope = dot(gφ, Δx)
                for _ in 1:20
                    if ϕ_kkt(x .+ α .* Δx; ρ=ρ) <= φx + b*α*slope
                        break
                    end
                    α *= cdec
                end
            end
            x .+= α .* Δx
            λ  += α * Δλ
            X   = hcat(X, x); push!(Λ, λ)
        end
        return X, Λ
    end
end


# ╔═╡ a61a63b7-716e-4500-9bc6-ab2caf9062e7
md"""
### Controls (KKT)

**x₁** $(@bind x1_kkt Slider(-3.5:0.1:1.5, default=-1.5, show_value=true))  
**x₂** $(@bind x2_kkt Slider(-3.0:0.1:3.0, default=-1.0, show_value=true))  

**λ₀** $(@bind λ0_kkt Slider(-2.0:0.1:2.0, default=0.0, show_value=true))  
**Iterations** $(@bind iters_kkt Select([3, 5, 8, 10, 15, 20]; default=8))  

**Method** $(@bind method_kkt Select([:gn, :full]; default=:gn))  
**Line search** $(@bind use_ls_kkt CheckBox(default=true))  

**ρ (merit weight)** $(@bind rho_kkt Slider(0.1:0.1:5.0, default=1.0, show_value=true))  

$(@bind run_kkt_btn Button("Run KKT"))
"""


# ╔═╡ 1fda2985-ab68-4460-98be-8621e5a5f1c8
let
    # === Run & plot (KKT section) ===
    run_kkt_btn   # recompute only when the button is clicked

    x0 = [x1_kkt, x2_kkt]
    X, Λ = kkt_solve_kkt(x0, λ0_kkt; iters=iters_kkt, method=method_kkt,
                         linesearch=use_ls_kkt, ρ=rho_kkt)

    fig = landscape_plot_kkt()
    plot_path_kkt!(fig, X)

    feas = abs(c_kkt(X[:,end]))
    stat = norm( ∇f_kkt(X[:,end]) .+ (Λ[end] .* ∇c_kkt(X[:,end])) )

    plt.annotate!(fig, -1.8, 15.3, plt.text("feas = $(round(feas,digits=4))", 9))
    plt.annotate!(fig, -1.8, 12.3, plt.text("stat = $(round(stat,digits=4))", 9))

    fig
end

# ╔═╡ ea0f0b1e-65b1-4c66-b9fa-7f2c428e3459
 

# ╔═╡ b4f6cce9-c39e-4325-a0e3-ab9c38179894
md"""
# Part IV - Log-domain Interior-Point Method (IPM): Tiny QP

We solve a 2D quadratic program with one inequality:

- Objective: minimize $f(x) = \tfrac{1}{2}(x - [1,0])^\top Q (x - [1,0])$ with $Q=\mathrm{diag}(0.5, 1)$.
- Constraint: $c(x) = (-1, 1)\cdot x - 1 \ge 0$ (i.e., $x_2 \ge x_1 + 1$).

**Log-domain substitution** for inequality handling:
$s = \sqrt{\rho}\,e^{\sigma}$, $\lambda = \sqrt{\rho}\,e^{-\sigma}$ so that $s\lambda = \rho$.

We solve the relaxed KKT system $r(z;\rho)=0$ with unknowns $z=[x;\sigma]$ using damped Newton plus Armijo backtracking on the merit function $\phi(z)=\tfrac{1}{2}\lVert r(z)\rVert^2$.
"""

# ╔═╡ 252554be-a839-4770-b222-fbb6a32df2ef
begin
    # Quadratic objective
    Q = Diagonal([0.5, 1.0])
    f(x)  = 0.5 * dot(x .- [1.0, 0.0], Q * (x .- [1.0, 0.0]))
    ∇f(x) = Q * (x .- [1.0, 0.0])

    # Single inequality c(x) ≥ 0 with gradient A = (-1, 1)
    A = [-1.0, 1.0]             # length-2 vector
    b = 1.0
    c(x) = dot(A, x) - b        # scalar
    J(x) = A'                   # 1×2 (shown for reference)
end

# ╔═╡ 62d85874-9673-4893-bbff-fa75b4462e96
begin
    # Confidence check: our analytic ∇f, ∇²f vs automatic differentiation
    fd_grad(x) = ForwardDiff.derivative(f, x)
    fd_hess(x) = ForwardDiff.derivative(fd_grad, x)

    xsamp = [-1.0, -0.25, 0.0, 0.5, 1.0]
    grad_err = maximum(abs.(fd_grad.(xsamp) .- ∇f.(xsamp)))
    hess_err = maximum(abs.(fd_hess.(xsamp) .- ∇²f.(xsamp)))
 
    (grad_err < 1e-10 && hess_err < 1e-10) ?
        tip(md"AD check passed: analytic derivatives match ForwardDiff on sample points.") :
        danger(md"AD check mismatch. Largest errors — grad: $(grad_err), hess: $(hess_err)")
end

# ╔═╡ 22bfe0a3-c61b-4dfe-8f20-a8bf807c2e14
begin
    # Regularized Newton direction (scalar LM-style)
    function newton_direction_reg(x; β0::Float64=1.0, max_tries::Int=20)
        g = grad_f(x)
        H = hess_f(x)
        if H > 0
            return -g / H, H
        end
        β = β0
        Hreg = H
        tries = 0
        while Hreg <= 0 && tries < max_tries
            Hreg = H + β
            β *= 2
            tries += 1
        end
        return -g / Hreg, Hreg
    end

    # Armijo backtracking (scalar)
    function armijo(f, x, Δx, g; b::Float64=1e-4, c::Float64=0.5, α0::Float64=1.0, max_backtracks::Int=50)
        α = α0
        fx = f(x)
        rhs = b * g * Δx
        for _ in 1:max_backtracks
            if f(x + α*Δx) <= fx + α*rhs
                return α
            end
            α *= c
        end
        return α
    end

    # Unified driver with full controls
    function newton_optimize(x0; tol=1e-8, maxiters=20, regularize::Bool=false, linesearch::Bool=false,
                             β0::Float64=1.0, b::Float64=1e-4, c::Float64=0.5, α0::Float64=1.0)
        x = float(x0)
        xhist = Float64[x]
        for _ in 1:maxiters
            g = grad_f(x)

            # Direction
            Δx, Hused = if regularize
                newton_direction_reg(x; β0=β0)
            else
                (-g / hess_f(x), hess_f(x))
            end

            if !isfinite(Δx) || abs(g) <= tol
                break
            end

            # Step length
            α = linesearch ? armijo(f, x, Δx, g; b=b, c=c, α0=α0) : 1.0

            x = x + α*Δx
            push!(xhist, x)
            if abs(grad_f(x)) <= tol
                break
            end
        end
        return xhist
    end
end

# ╔═╡ 858d4ee1-4f15-4ced-b984-c0291237d359
begin
    # Build plots for whichever methods are selected
    plots = Plots.Plot[]  # vector of subplots

    if show_plain
        xhist_plain = newton_optimize(x0_cmp; maxiters=iters_cmp, regularize=false, linesearch=false)
        push!(plots, plot_trace(xhist_plain; title="Plain"))
    end
    if show_reg
        xhist_reg   = newton_optimize(x0_cmp; maxiters=iters_cmp, regularize=true,  linesearch=false, β0=beta0)
        push!(plots, plot_trace(xhist_reg; title="Regularized (β₀=$(round(beta0,digits=2)))"))
    end
    if show_rls
        xhist_rls   = newton_optimize(x0_cmp; maxiters=iters_cmp, regularize=true,  linesearch=true,
                                      β0=beta0, b=armijo_b, c=armijo_c)
        push!(plots, plot_trace(xhist_rls; title="Reg + LineSearch (b=$(armijo_b), c=$(round(armijo_c,digits=2)))"))
    end

    if isempty(plots)
        md"Select at least one method to visualize."
    else
        n = length(plots)
        layout_val = orientation == "row (1×N)" ? (1,n) : (n,1)
        plt.plot(plots..., layout=layout_val, size=(1200, orientation=="row (1×N)" ? 360 : 900))
    end
end


# ╔═╡ 82e2eca9-4991-4538-8e1f-455cd46f849f
begin
    # Generic version
    function landscape_plot(; xlim=(-4,4), ylim=(-4,4), nsamp=120, show_colorbar=false)
        xs = range(xlim[1], xlim[2], length=nsamp)
        ys = range(ylim[1], ylim[2], length=nsamp)
        Z  = [ f([x,y]) for x in xs, y in ys ]  # size(Z) = (length(xs), length(ys))

        p = plt.contour(xs, ys, Z;
                        levels=18,
                        colorbar=show_colorbar,
                        xlabel="x₁", ylabel="x₂",
                        title="Objective contours & constraint",
                        legend=:topleft,
                        aspect_ratio=:equal,
                        xlims=(xlim[1], xlim[2]),  # hard-lock axes
                        ylims=(ylim[1], ylim[2]))

        xc = collect(xs)
        yc = @. xc^2 + 2xc
        plt.plot!(p, xc, yc; lw=2, label="c(x)=0")
        plt.scatter!(p, [xstar[1]], [xstar[2]]; marker=:star5, ms=8, label="x⋆")

        # re-enforce limits after adding series (paranoid, but safe)
        plt.xlims!(p, (xlim[1], xlim[2]))
        plt.ylims!(p, (ylim[1], ylim[2]))
        return p
    end

    plot_path!(p, X) = begin
        plt.plot!(p, X[1,:], X[2,:]; marker=:x, ms=6, lw=1.5, label="iterates")
        plt.xlims!(p, (-4,4)); plt.ylims!(p, (-4,4))   # keep locked if path goes outside
        p
    end
end


# ╔═╡ db230e89-e8ea-4c57-8517-0dbf68cab6b7
md"""
## Visualizing the objective and the constraint

We plot contour lines of $f(x)$ and the line $c(x)=0$ (the feasible region is on or above that line, i.e., $c(x)\ge 0$).
"""

# ╔═╡ d9c51abd-2488-405f-9b4c-799f496c0dd1
begin
    function plot_landscape(; xlim=(-4.0, 4.0), ylim=(-4.0, 4.0))
        xs = range(xlim[1], xlim[2], length=200)
        ys = range(ylim[1], ylim[2], length=200)
        Z  = [f([xi, yi]) for yi in ys, xi in xs]

        p = plt.contour(
            xs, ys, Z;
            fill=false,
            xlabel="x₁", ylabel="x₂",
            title="Objective & constraint",
            aspect_ratio=1,          # square figure; keep if you like
            xlims=xlim, ylims=ylim   # << fix the visible ranges
        )

        plt.plot!(p, xs, xs .+ 1; label="c(x)=0 (x₂ = x₁ + 1)")
        return p
    end

    fig_test = plot_landscape()  # will be limited to [-4,4] × [-4,4]
end


# ╔═╡ af7cf190-fa37-40f7-ac9e-dbdb8e72fff8
md"""
## Log-domain IP residuals

With $z=[x;\sigma]$ and parameter $\rho>0$, define
$\lambda(z,\rho)=\sqrt{\rho}\,e^{-\sigma}$ and $s(z,\rho)=\sqrt{\rho}\,e^{\sigma}$.

Residuals:
- Stationarity (2 numbers): $\nabla f(x) - \lambda A = 0$.
- Primal feasibility (1 number): $c(x) - s = 0$.

So $r(z;\rho) = \begin{bmatrix} \nabla f(x) - \lambda A \\ c(x) - s \end{bmatrix} \in \mathbb{R}^3$.
"""

# ╔═╡ 6c5b33c5-94fc-4b41-8128-696a58ca9b64
begin
    function ip_residual(z::AbstractVector, ρ::Real)
        x = z[1:2]
        σ = z[3]
        λ = sqrt(ρ) * exp(-σ)      # ≥ 0
        s = sqrt(ρ) * exp(σ)       # ≥ 0
        r1 = ∇f(x) .- (λ .* A)     # length-2
        r2 = c(x) - s              # scalar
        return vcat(r1, r2)        # length-3
    end
end

# ╔═╡ 64d20ba5-d195-41ef-88b8-d2968bf87ac2
md"""
## Newton solve for fixed $\rho$

We solve $r(z; \rho)=0$ with damped Newton on the merit $\phi(z)=\tfrac{1}{2}\lVert r(z)\rVert^2$.

Directional derivative of $\phi$ along $\Delta z$ is $r(z)^\top J(z)\,\Delta z$ where $J$ is the Jacobian of $r$.
"""

# ╔═╡ 4f9c5ddd-05f5-4826-a775-bc1af4555153
begin
    function newton_solve(z0::AbstractVector, ρ; tol=1e-10, maxiter=50, α_min=1e-12)
        z = copy(z0)
        R = ip_residual(z, ρ)
        Zhist = reshape(z, :, 1)

        for k in 1:maxiter
            if norm(R) ≤ tol
                return z, Zhist
            end
            Jz = ForwardDiff.jacobian(zz -> ip_residual(zz, ρ), z)  # 3×3
            Δz = -Jz \ R

            ϕz  = 0.5 * dot(R, R)
            gΔ  = dot(R, Jz * Δz)       # directional derivative of ϕ at z along Δz
            α   = 1.0
            z_new = z .+ α .* Δz
            R_new = ip_residual(z_new, ρ)
            ϕnew  = 0.5 * dot(R_new, R_new)
            c_armijo, shrink = 1e-4, 0.5

            if gΔ > 0
                # fallback: just ensure decrease in ϕ
                while ϕnew > ϕz && α > α_min
                    α *= shrink
                    z_new = z .+ α .* Δz
                    R_new = ip_residual(z_new, ρ)
                    ϕnew  = 0.5 * dot(R_new, R_new)
                end
            else
                # standard Armijo
                while ϕnew > ϕz + c_armijo*α*gΔ && α > α_min
                    α *= shrink
                    z_new = z .+ α .* Δz
                    R_new = ip_residual(z_new, ρ)
                    ϕnew  = 0.5 * dot(R_new, R_new)
                end
            end

            z, R = z_new, R_new
            Zhist = hcat(Zhist, z)
        end
        return z, Zhist
    end
end

# ╔═╡ 1fb54858-898e-4d0d-8435-746bf3c17e65
md"""
#### KKT sanity checks to report: 
* stationarity norm $\lVert\nabla f(x) - \lambda A\rVert$
* primal feasibility $c(x)$ 
* dual feasibility $\lambda$ 
* and complementarity slackness $\lambda\,c(x)$

These should approach zero feasibility violation and complementarity as $\rho \to 0$.
"""


# ╔═╡ 2070905c-d96b-4e3d-8a51-f0bf26ba98b8
begin
    function kkt_checks(z, ρ)
        x, σ = z[1:2], z[3]
        λ = sqrt(ρ) * exp(-σ)
        (; stationarity_norm = norm(∇f(x) .- (λ .* A)),
           primal_feas       = c(x),     # should be ≥ 0
           dual_feas         = λ,        # should be ≥ 0
           complementarity   = λ * c(x))
    end
end


# ╔═╡ 08270f19-7c1c-4707-baf8-fa985519baf3
md"""
## Demo: follow the central path by decreasing $\rho$

We warm-start Newton at a smaller $\rho$ and plot the iterates on the $(x_1,x_2)$ plane.
"""


# ╔═╡ a4c50b10-0a6f-4967-9895-e5171f44fafa
begin
    # Initial guess (must be feasible or strictly interior for the barrier)
    x0_init = [-2.0, 2.0]
    σ0_init = 0.0
    z0_init = vcat(x0_init, σ0_init)   # [x; σ] ∈ ℝ^3

    # Central-path parameters
    ρ_path1 = 1.0
    ρ_path2 = 1e-8
end

# ╔═╡ 71530ac2-841c-4616-98e3-2dedd63e3834
begin
    # cold-start both runs from the same z0_init
    sol_rho1_cold, traj_rho1_cold = newton_solve(z0_init, ρ_path1; tol=1e-10)
    sol_rho2_cold, traj_rho2_cold = newton_solve(z0_init, ρ_path2; tol=1e-10)

    fig_ipm_compare = plot_landscape(xlim=(-4,4), ylim=(-4,4))
    plt.plot!(fig_ipm_compare; legend=:bottomright, grid=true)

    # starting point
    plt.scatter!(fig_ipm_compare, [z0_init[1]], [z0_init[2]];
        label="start", markershape=:star5, markersize=9, markerstrokewidth=0, color=:black)

    # path & final for ρ = 1.0
    plt.plot!(fig_ipm_compare, vec(traj_rho1_cold[1, :]), vec(traj_rho1_cold[2, :]);
        label="iterates ρ=1.0", lw=2, marker=:circle, markersize=3, color=:blue) 

    # path & final for ρ = 1e-8
    plt.plot!(fig_ipm_compare, vec(traj_rho2_cold[1, :]), vec(traj_rho2_cold[2, :]);
        label="iterates ρ=1e-8", lw=2, marker=:utriangle, markersize=3, color=:red)
    

    fig_ipm_compare
end



# ╔═╡ fe6ec86b-84c1-4e82-85e0-dce1a316214a
md"""
### Optional: Jacobian of the residual at the final point

The Jacobian $J(z)$ of $r(z;\\rho)$ at the final iterate can be inspected via eigenvalues to gauge local conditioning.
"""

# ╔═╡ 686b3f99-11df-4cc0-b739-3ff8476d217d


# ╔═╡ 8aa33896-2585-4a60-86dc-7386b96feb9f


# ╔═╡ 3132cda5-ad6e-4ae2-afba-ea18d09cf8df
md"""
# Part V - Sequential Quadratic Programming (SQP)

**Idea.** Solve a nonlinear constrained problem by repeatedly solving a **quadratic program (QP)** built from local models.

- Quadratic model of the Lagrangian/objective near the current iterate.
- Linearize the constraints around the current iterate.
- Each iteration: solve a QP to get a step $d$, then update $x \leftarrow x + \alpha d$.
- Strengths: Newton-like local convergence (often superlinear), warm-start friendly.
"""



# ╔═╡ 74d5c04f-3f63-41ba-818d-ce217cd18022
md"""
## Problem & KKT recap

We consider
$\min_{x \in \mathbb{R}^n} \ f(x) \quad \text{s.t.}\quad g(x)=0,\ \ h(x)\le 0$.

At a candidate optimum $x^\star$, the KKT conditions require multipliers
$\lambda \in \mathbb{R}^m$, $\mu \in \mathbb{R}^p_{\ge 0}$ such that
$\nabla f(x^\star) + \nabla g(x^\star)^\top \lambda + \nabla h(x^\star)^\top \mu = 0$,
$g(x^\star)=0,\ \ h(x^\star)\le 0,\ \ \mu \ge 0,\ \ \mu \odot h(x^\star)=0$.
"""


# ╔═╡ 4b560ea8-8104-40ce-bf44-b5c2267132eb
md"""
## Local models that define the QP

At iterate $x_k$ with multipliers $(\lambda_k,\mu_k)$:

**Quadratic model**
$m_k(d) \;=\; \nabla f(x_k)^\top d \;+\; \tfrac{1}{2}\, d^\top B_k\, d,$
with $B_k \approx \nabla^2_{xx}\mathcal{L}(x_k,\lambda_k,\mu_k)$. Common choices: exact Hessian, (L-)BFGS, or Gauss–Newton.

**Linearized constraints**
$ g(x_k) + \nabla g(x_k)\, d = 0, \qquad h(x_k) + \nabla h(x_k)\, d \le 0.$
"""

# ╔═╡ 96738f09-e894-499e-9e30-15692e2434dd
md"""
## The SQP subproblem (QP)

$
\begin{aligned}
\min_{d \in \mathbb{R}^n}\quad &
\nabla f(x_k)^\top d \;+\; \tfrac{1}{2}\, d^\top B_k\, d \\
\text{s.t.}\quad &
\nabla g(x_k)\, d + g(x_k) = 0, \\
&
\nabla h(x_k)\, d + h(x_k) \le 0.
\end{aligned}
$

Solving this QP yields a step $d_k$ and QP multipliers $(\lambda_{k+1},\mu_{k+1})$.
Update with $x_{k+1} = x_k + \alpha_k d_k$ (line search or trust region).
"""



# ╔═╡ 704f8f36-fda1-445e-9551-942fd85bff42
md"""
## SQP (line-search flavor): 6 steps

1. Initialize $x_0$, multipliers $(\lambda_0,\mu_0)$, and $B_0 \succ 0$.
2. Build the QP at $x_k$ using $B_k$, $\nabla g(x_k)$, $\nabla h(x_k)$, $g(x_k)$, $h(x_k)$.
3. Solve the QP $\Rightarrow$ get $d_k$ and $(\lambda_{k+1},\mu_{k+1})$.
4. Choose $\alpha_k \in (0,1]$ via globalization (merit or filter).
5. Set $x_{k+1} = x_k + \alpha_k d_k$.
6. Update $B_{k+1}$ (e.g., damped BFGS). Stop when KKT residuals are small.
"""



# ╔═╡ b6acef39-d40d-42e6-b037-10713dc1254e
md"""
### Globalization: make SQP robust

**Merit (penalty) function** for line search, e.g.
$\phi(x) \;=\; f(x) \;+\; \tfrac{\rho}{2}\,\|g(x)\|^2 \;+\; \rho\,\|h(x)_+\|_1,$
and choose $\alpha_k$ by Armijo/backtracking so $\phi$ decreases.

**Filter methods** accept steps that improve *either* objective *or* feasibility.

**Trust-region SQP**: restrict $\|d\|\le \Delta_k$, compare predicted vs actual reduction, adjust $\Delta_k$.

### Inequalities & active sets (intuition)

- The QP contains the **linearized inequalities** $h(x_k)+\nabla h(x_k)d \le 0$.
- Its KKT system enforces complementarity via multiplier signs and active constraints.
- A small **working set** (estimated active constraints) tends to stabilize across iterations, enabling warm starts and fast solves.
 
"""



# ╔═╡ 34737fc8-095c-4061-8661-e776c23c7eed


# ╔═╡ fc35e75c-b81d-44eb-bcee-74c590e38652


# ╔═╡ 32587054-655f-4f06-9ff8-500ba44bec76


# ╔═╡ 3dfc4b4e-85e3-4dee-ae33-5ddb2a4eec29
question_box(md"hello")

# ╔═╡ e8f62342-88ab-4754-af33-e2347be2daa0
Foldable(md"Compressed vs. separated form", md"
We can either express the values of $x$ and $u$ at $t_{k + \frac{1}{2}}$ with the expressions above, or we can set them as decision variables, and enforce the expressions above as equality constraints.
Doing the former results in *compressed form* and the latter *separated form*.
The compressed form tends to perform better with a large number of segments and the separated form a small number of segments [^kelly2017].
")

# ╔═╡ Cell order:
# ╟─81ebc291-89f0-4c1e-ac34-d5715977dd86
# ╟─9543f7bc-ab36-46ff-b471-9aa3db9739e4
# ╠═8969e78a-29b0-46d3-b6ba-59980208fe5b
# ╟─d90e9be0-7b68-4139-b185-6cbaad0d307e
# ╠═7b896268-4336-47e2-a8b5-f985bfde51f5
# ╟─342decc1-43fa-432a-9a9c-757a10ba6a5d
# ╟─fd1ad74b-cb74-49a7-80b8-1a282abfdff2
# ╠═49d5b2e6-eb29-478c-b817-8405d55170b1
# ╠═950c61b8-f076-4b9a-8970-e5c2841d75f2
# ╠═92841a2e-bc0d-40f8-8344-a5c398a67275
# ╠═8813982c-8c9a-4706-91a8-ebadf9323a4f
# ╠═4307a2f3-0378-4282-815b-9ed1fa532e1c
# ╠═a45ed97f-f7c1-4ef5-9bc7-654e827f751b
# ╠═5a17f83e-751b-4244-9c15-7165645bfe29
# ╟─12c03202-58b2-482e-a65c-b83bc1f6eed1
# ╠═662d58d7-4c9c-4699-a846-cb6070c507d9
# ╠═27760607-b740-47dc-a810-c332baa2bd2d
# ╟─7765e032-c520-4f97-b877-0d479f383f28
# ╠═a098a49b-a368-4929-824c-385a06b88696
# ╠═62d85874-9673-4893-bbff-fa75b4462e96
# ╠═a62f1b6a-87fe-4401-bc13-42166ca0e129
# ╠═b2ddc048-4942-43bc-8faa-1921062d8c9c
# ╠═6fd7a753-6db7-4c37-9e22-dc63dd3072c8
# ╠═49e0f5c3-fe14-42e1-9b3a-83c4447148a8
# ╠═096ebc95-f133-4ca3-b942-cf735faaa42b
# ╠═fdf41c76-4405-49e0-abfa-5c5193de99f4
# ╠═22bfe0a3-c61b-4dfe-8f20-a8bf807c2e14
# ╠═d259c1b8-3716-4f80-b462-3b3daebb444d
# ╠═858d4ee1-4f15-4ced-b984-c0291237d359
# ╠═e012d7e0-0181-49d0-bb78-995378c4f87a
# ╠═0a3eb748-fab3-4f8b-993e-2246c32fb6aa
# ╠═3989b8e1-ac1f-430d-9d72-e298ba7ae0ca
# ╠═e466ffb2-fcc8-4c3b-9059-7fd68a4265f2
# ╠═09a301c7-c1f4-4890-afe9-fbc1d7c3b905
# ╠═53020b8a-4948-467a-8629-bef9496d0374
# ╠═26c887ce-d95b-4e38-9717-e119de0e80ca
# ╟─a1b7f614-710a-4ce8-815b-2f94754088c4
# ╟─6fd82e73-6fef-4f7f-b291-f94cbac0d268
# ╠═52718e0b-f958-445e-b9b6-9e5baf09e81a
# ╠═edce5b27-9af8-4010-9d9f-60681b2f427c
# ╠═c139ccc9-0355-4c41-8d82-0c97fef50900
# ╟─43c5ccb9-d51e-4d73-b7be-f8b9dd599130
# ╠═e0a7f431-61dd-4df2-b53c-d81c7a302baf
# ╠═82e2eca9-4991-4538-8e1f-455cd46f849f
# ╠═3e3d6e18-f922-4e91-8eaf-bc26b8f91757
# ╠═31407c3f-8f3d-4dcc-bc49-cedd8ca14013
# ╠═a61a63b7-716e-4500-9bc6-ab2caf9062e7
# ╠═1fda2985-ab68-4460-98be-8621e5a5f1c8
# ╠═ea0f0b1e-65b1-4c66-b9fa-7f2c428e3459
# ╠═b4f6cce9-c39e-4325-a0e3-ab9c38179894
# ╠═252554be-a839-4770-b222-fbb6a32df2ef
# ╠═db230e89-e8ea-4c57-8517-0dbf68cab6b7
# ╠═d9c51abd-2488-405f-9b4c-799f496c0dd1
# ╠═af7cf190-fa37-40f7-ac9e-dbdb8e72fff8
# ╠═6c5b33c5-94fc-4b41-8128-696a58ca9b64
# ╠═64d20ba5-d195-41ef-88b8-d2968bf87ac2
# ╠═4f9c5ddd-05f5-4826-a775-bc1af4555153
# ╠═1fb54858-898e-4d0d-8435-746bf3c17e65
# ╠═2070905c-d96b-4e3d-8a51-f0bf26ba98b8
# ╠═08270f19-7c1c-4707-baf8-fa985519baf3
# ╠═a4c50b10-0a6f-4967-9895-e5171f44fafa
# ╠═71530ac2-841c-4616-98e3-2dedd63e3834
# ╠═fe6ec86b-84c1-4e82-85e0-dce1a316214a
# ╠═686b3f99-11df-4cc0-b739-3ff8476d217d
# ╠═8aa33896-2585-4a60-86dc-7386b96feb9f
# ╠═3132cda5-ad6e-4ae2-afba-ea18d09cf8df
# ╟─74d5c04f-3f63-41ba-818d-ce217cd18022
# ╟─4b560ea8-8104-40ce-bf44-b5c2267132eb
# ╟─96738f09-e894-499e-9e30-15692e2434dd
# ╟─704f8f36-fda1-445e-9551-942fd85bff42
# ╟─b6acef39-d40d-42e6-b037-10713dc1254e
# ╠═34737fc8-095c-4061-8661-e776c23c7eed
# ╠═fc35e75c-b81d-44eb-bcee-74c590e38652
# ╠═32587054-655f-4f06-9ff8-500ba44bec76
# ╠═3dfc4b4e-85e3-4dee-ae33-5ddb2a4eec29
# ╠═e8f62342-88ab-4754-af33-e2347be2daa0
