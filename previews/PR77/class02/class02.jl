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
# Overview of Chapter 2

Lecture 2 is the course’s optimization backbone: it makes explicit the idea that most control problems are optimization problems in disguise. 
We set the common language (gradients/Hessians, KKT systems, globalization) and the “solver toolbox” (penalty, augmented Lagrangian, primal–dual interior-point, SQP) that shows up everywhere else: 
* MPC is a QP/NLP solved online; 
* trajectory optimization is an NLP with sparse structure; 
* distributed control leans on operator splitting; 

and modern differentiable controllers backprop through these solvers. 

The point of this chapter is to give you a reliable recipe for turning a control task into a well-posed QP/NLP and picking/configuring a solver that actually converges.

Positionally, this lecture is the hinge between “dynamics & modeling” (Class 1) and everything that follows: PMP/LQR reframed via KKT and Riccati; nonlinear trajectory optimization and collocation (Class 5); distributed MPC/ADMM (Class 8); GPU-accelerated solves (Class 9); and the learning side—adjoints for Neural DEs (Class 10) and optimization-as-a-layer for PINNs/neural operators (Classes 11–13). 

##### General structure of chapter (please refer to lecture slides as needed!)

- **Root finding:** how implicit time-stepping (Backward Euler) turns into solving $r(x)=0$ each step.
- **Unconstrained minimization:** Newton’s method and why we need *globalization strategies* (regularization + line search).
- **Constrained minimization:** KKT conditions in action for equality constraints.
- **Interior Point Method:** IPM (check lecture slides for augmented lagrangian method and other penalty methods).
- **SQP (Sequential Quadratic Programming):** how to reduce a nonlinear program to a sequence of easy QPs.
""" 

# ╔═╡ fd1ad74b-cb74-49a7-80b8-1a282abfdff2
md"""
# Part I — Unconstrained Minimization as Root Finding

Many steps in control are solutions of **nonlinear equations** $r(x)=0$:
- **Implicit simulation (Backward Euler)** each step solves $r(x)=0$.
- **Newton steps** inside optimizers solve a linearized $r=0$.
- **Adjoints** are solutions of linear(ized) equations.

We’ll keep one reusable pattern: define a residual $r(x)$, compute its Jacobian ($\partial r(x)$), and choose an iteration that drives $\|r(x_k)\|$ down quickly and reliably. 

**Going from implicit step to residual**

For $\dot x=f(x)$, one Backward Euler step with step size $h$ satisfies

$x_{n+1} = x_n + h\,f(x_{n+1})$

Move everything to the left to get the **residual**

$r(x) \;\equiv\; x_n + h\,f(x) - x,
\qquad
\partial r(x) \;=\; h\,J_f(x) - I$

Two important facts:
- For small $h$, $\partial r \approx -I$ ⇒ Newton systems are well-conditioned.
- Note $J_f$ carries physics; using it (rather than a crude approximation) is what makes Newton fast.

We compare two solvers for $r(x)=0$: a **fixed-point** iteration and **Newton’s method**.  

**Fixed-point (Picard).** Define $g(x)=x_n+h\,f(x)$ and iterate

$x_{k+1}=g(x_k)$

It converges if $g$ is a contraction near the root (intuitively $h\|J_f\|$ small). Rate: **linear**.

**Newton’s method.** Linearize $r$ and solve for a correction:

$(\partial r(x_k))\,\Delta x_k = -\,r(x_k), \qquad x_{k+1}=x_k+\Delta x_k$

Near a solution, the rate is **quadratic**. 
"""

# ╔═╡ 5e6c4ea7-b283-423c-b9c0-421d53cebc2d
md"""
We now will move onto a small example to use root finding on.

**Reminder of the pendulum dynamics:** we use the simple, undamped pendulum with state $x=[\theta,\;v]$ so $\dot{\theta}=v$ and $\dot{v}=-(g/\ell)\sin\theta$, where $g=9.81\,\mathrm{m/s^2}$ and $\ell$ is the length.  
In vector form, $f(x)=\begin{bmatrix}v\\ -(g/\ell)\sin\theta\end{bmatrix}$, which is what `pendulum_dynamics` returns.
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
For the implicit step, define the residual and jacobian as
$r(x) = x_n + h\,f(x) - x,$
$\partial r(x) = \dfrac{\partial}{\partial x}\big(x_n + h\,f(x) - x\big) = h\,J_f(x) - I.$
"""


# ╔═╡ 80035b9c-eba6-469c-b138-c6c792979493
md"""
Backward Euler converts the implicit update into a root-finding task: the residual $r(x)=x_n+h\,f(x)-x$ measures how far a candidate $x$ misses the equation $x=x_n+h\,f(x)$, and the step is the root $r(x)=0$.  
Its Jacobian is $\partial r(x)=h\,J_f(x)-I$, which is close to $-I$ when $h$ is small—so Newton steps are well-conditioned while $J_f$ carries the system’s physics.
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
Now we can move onto to actually implementing a root finding solver 

Each Newton update solves
$(h\,J_f(x_k)-I)\,\Delta x_k = -\,r(x_k).$
For small $h$, the matrix is close to $-I$ (well-conditioned); as $h$ grows or the dynamics stiffen, conditioning deteriorates — that is when damping and scaling matter.
 
**Fixed point (Picard)** updates: $x_{k+1} = g(x_k) = x_n + h\,f(x_k)$.  
Converges locally if \(g\) is a contraction (roughly, small enough \(h\)).
 
"""

# ╔═╡ 4307a2f3-0378-4282-815b-9ed1fa532e1c
begin 

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
#####  How to use the root finding demo

1. Pick an initial state $x_n=[\theta, v]$ and a step size $h$.
2. Click **Run** to see $\|r(x_k)\|$ vs iteration (log scale).
3. Try increasing $h$:
   - Fixed-point slows down or fails (not a contraction).
   - Newton stays fast as long as $\partial r=hJ_f-I$ is nonsingular.

**Good starting point to try:** $\theta\in[0.1,0.3]$, $v=0$, $h=0.1$.

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


# ╔═╡ 56c965c9-5acc-40a5-b1dd-c3a59f0462a9
md"""
**What you should see:** 

(For $\theta=0.15, v=0.0, h = 0.1$) Newton drives the residual from ~$10^{-3}$ to ~$10^{-11}$ in about 3 iterations, while fixed-point takes $~15$ iterations to reach only ~$10^{-8}$. The steep, almost vertical drop of the orange curve (log scale) is the signature of **near-quadratic** convergence; the blue curve decays roughly linearly and can stagnate. Because Newton uses the true Jacobian $h J_f - I$, this advantage usually grows as $h$ (or stiffness) increases.  
"""

# ╔═╡ af82d16d-c649-461b-856a-42355517d9f4
md"""
##### From roots to minima (part II of chapter)

Minimization often reduces to **root finding** on the gradient:

$\min_x f(x)\quad \Longleftrightarrow \quad \nabla f(x)=0$

The next section of this chapter takes the same Newton machinery you just used on $r=0$ and applies it to $\nabla f=0$, then adds **globalization** (regularization + line search) to make it behave well away from the optimum.
"""


# ╔═╡ 662d58d7-4c9c-4699-a846-cb6070c507d9
md"""
# Part II - Unconstrained Minimization — Newton, Regularization, and Line Search

In this next part we will apply Newton's method to a nonconvex scalar objective. We will see that that plain Newton (i.e without additional safeguards) can **increase** the objective (negative curvature). To handle this we will go over a technique to stabilize newton's method using **Hessian regularization**. Finally to assure robust progress throughout the solution process and avoiding oscillations, we will go also show how to add **Armijo backtracking**.

We will work with a toy scalar function which has multiple critical points (minima/maxima) which makes it a nice testbed to show where Newton can go wrong. The function is  

$f(x) = x^4 + x^3 - x^2 - x$

Before doing anything related to newton's method, let's just plot the actual function we have and visualize it! 
"""

# ╔═╡ 27760607-b740-47dc-a810-c332baa2bd2d
begin
 
    f(x::Real)  = x^4 + x^3 - x^2 - x
    ∇f(x::Real) = 4x^3 + 3x^2 - 2x - 1
    ∇²f(x::Real)= 12x^2 + 6x - 2

    grad_f(x) = ∇f(x)
    hess_f(x) = ∇²f(x)
	
    function plot_fx(; xrange=(-1.75, 1.25), npts=1000, title="Objective")
        xs = range(xrange[1], xrange[2], length=npts)
        ys = f.(xs)
        ylo, yhi = extrema(ys)
        pad = 0.05*(yhi - ylo + eps())  # small padding
        plt.plot(xs, ys;
            lw=2, label="f(x)", xlabel="x", ylabel="f(x)", title=title,
            xlims=xrange, ylims=(ylo - pad, yhi + pad)
        )
    end

    function plot_trace(xhist; xrange=(-1.75, 1.25), npts=1000, title="",
                        show_first_iter::Bool=true, show_first_model::Bool=true)

        xs = range(xrange[1], xrange[2], length=npts)
        ys = f.(xs)
        ylo, yhi = extrema(ys)
        pad = 0.05*(yhi - ylo + eps())

        p = plt.plot(xs, ys;
            lw=2, label="f(x)", xlabel="x", ylabel="f(x)", title=title,
            xlims=xrange, ylims=(ylo - pad, yhi + pad)
        )
 
        x_in = [x for x in xhist if isfinite(x) && xrange[1] ≤ x ≤ xrange[2]]
        if !isempty(x_in)
            plt.scatter!(p, x_in, f.(x_in); marker=:x, ms=7, label="iterates")
        end
 
        if show_first_iter && length(xhist) ≥ 2
            x0, x1 = xhist[1], xhist[2]
            if isfinite(x0) && isfinite(x1) &&
               (xrange[1] ≤ x0 ≤ xrange[2]) && (xrange[1] ≤ x1 ≤ xrange[2]) &&
               isfinite(f(x0)) && isfinite(f(x1))
                plt.plot!(p, [x0, x1], [f(x0), f(x1)];
                    lw=2, ls=:dash, label="first step x₀ → x₁"
                )
                plt.scatter!(p, [x1], [f(x1)]; marker=:diamond, ms=8, label="x₁")
            end
        end
 
        if show_first_model && !isempty(xhist) && isfinite(xhist[1])
            x0 = xhist[1]
            g0, H0 = grad_f(x0), hess_f(x0)
            if isfinite(g0) && isfinite(H0)
                m0(x) = f(x0) + g0*(x - x0) + 0.5*H0*(x - x0)^2
                width = 0.75
                xs_loc = range(max(xrange[1], x0 - width), min(xrange[2], x0 + width), length=250)
                if length(xs_loc) > 1
                    plt.plot!(p, xs_loc, m0.(xs_loc); ls=:dot, lw=2, label="quadratic model @ x₀")
                end
            end
        end

        return p
    end
end

# ╔═╡ a098a49b-a368-4929-824c-385a06b88696
plot_fx(title="f(x) = x^4 + x^3 - x^2 - x")

# ╔═╡ 17ac372e-87e6-4649-ba9d-1df1cdb7b55b
md"""
We can see that this function has a global minima on the right side of the plot close to $x_1\approx0.6$ whilst it has a local minima around $x_1\approx-1$. This is a perfect example as we can analyze what happens when we start in the basin of attraction of the local minima on the left. Now let's go over how the newton method should be implemented
"""

# ╔═╡ a62f1b6a-87fe-4401-bc13-42166ca0e129
md"""
#### Plain Newton update

We want a **stationary point** of $f$, i.e. solve

```math
\nabla f(x) = 0
```

Newton’s method does this by **linearizing the gradient** at the current point ($x_k$):

```math
\nabla f(x) \approx \nabla f(x_k) + \nabla^2 f(x_k)\,(x - x_k)
```

Setting this approximation to zero and solving for (x) gives the update

```math
x_{k+1} = x_k - \big[\nabla^2 f(x_k)\big]^{-1}\,\nabla f(x_k)
```

In **one dimension**, the Hessian is just the scalar ($f''(x_k)$), so

```math
\boxed{\,x_{k+1} = x_k - \dfrac{f'(x_k)}{f''(x_k)}\,}
```

**Quadratic-model view (same update).** Around (x_k), approximate (f) by the quadratic

```math
m_k(s)= f(x_k)+ \nabla f(x_k)\, s + \tfrac12\, \nabla^2 f(x_k)\, s^2 .
```

Minimizing ($m_k$) w.r.t. ($s$) gives ($s_k = -\nabla f(x_k)/\nabla^2 f(x_k)$) (1D), hence the same update ($x_{k+1}=x_k+s_k$).
This reveals a key fact:

* If ($f''(x_k) > 0$) (positive curvature), then ($s_k$) is a **descent direction** and the step tends to **decrease** (f).
* If ($f''(x_k) < 0$) (negative curvature), the local quadratic is **concave** and the same formula points **uphill**—plain Newton can **increase** (f).

In higher dimensions the descent test is

```math
\nabla f(x_k)^\top \big[\nabla^2 f(x_k)\big]^{-1}\nabla f(x_k) > 0,
```

which holds when the Hessian is positive definite; if the Hessian is indefinite/negative, Newton’s direction need not be descent.
""" 

# ╔═╡ be41026f-cb28-4647-8db6-1d243739f444
md"""
The following code implements an iterative optimization algorithm using Newton for our 1D nonconvex scalar function (no safeguards yet)
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


# ╔═╡ 1ffa2941-619b-400a-ba0f-56baa6ee7f59
md"""
The first slider corresponds to the initial starting pointer iterate for the newton's method. The second slider corresponds to the number of iterations we should run the algorithm for.

First test out the algorithm with an initial start position at $x=-1.5$ and $5$ iterations. The algorithm will converge to a local minima at $-1$.
"""

# ╔═╡ 6fd7a753-6db7-4c37-9e22-dc63dd3072c8
begin
    @bind x0_plain PlutoUI.Slider(-1.75:0.01:0.5, default=-1.50, show_value=true)
end


# ╔═╡ 49e0f5c3-fe14-42e1-9b3a-83c4447148a8
begin
    @bind iters_plain PlutoUI.Slider(1:20, default=8, show_value=true)
end

# ╔═╡ 096ebc95-f133-4ca3-b942-cf735faaa42b
begin
    xhist_plain_1 = newton_optimize_plain(x0_plain; maxiters=iters_plain)
    plot_trace(
        xhist_plain_1;
        title = "Plain Newton from x₀ = $(round(x0_plain,digits=3)) and iters = $(iters_plain))",
        show_first_iter = true,
        show_first_model = true,   # set false if you only want the step segment
    )
end

# ╔═╡ fdf41c76-4405-49e0-abfa-5c5193de99f4
md"""
To avoid issues with newton, we add globalization strategies which include regularization and Armijo backtracking

Two simple fixes:

1. **Regularization:** If $\nabla^2 f(x_k)\le 0$, replace it by $\nabla^2 f(x_k) + \beta $ with $\beta>0$, so the step is a **descent** direction.
2. **Armijo line search:** Scale the step by $\alpha \in (0,1]$ until
$f(x_k+\alpha\Delta x) \le f(x_k) + b\,\alpha\,\nabla f(x_k)\,\Delta x$
with typical $b=10^{-4}$, $c=0.5$ for backtracking.

We’ll expose both as toggles.
"""


# ╔═╡ d259c1b8-3716-4f80-b462-3b3daebb444d
md"""
#### Controls
Use the controls to find settings where it's clear that the left/last plot is the best. I.e the use of regularization and backtracking ensures we move towards the true optimal minima with no over-shooting due to backtracking. A good example is a starting point of roughly -0.36 and $\beta_0=1$ and $c=0.5$. In that setting we can see that both only the plain newton cannot solve to optimality. The pure regularized version has oscillation as it overshoots the minima. On the other hand the regularized and line search implementation is able to find the global minima efficiently.

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


# ╔═╡ 76c51e32-bd0e-4e72-8c47-64352da13d3e
md"""
If you are interested in seeing how the plotting works, open the code for the cell below.
"""

# ╔═╡ e012d7e0-0181-49d0-bb78-995378c4f87a
md"""
Here are the main takeaways for part II.

- **Plain Newton** is fast **if** you’re in a nice region and the Hessian is positive.
- **Regularization** turns the Newton step into a descent direction when curvature is negative or near-singular.
- **Armijo backtracking** fixes overshooting and makes progress predictable without hand-tuning step sizes.
- These building blocks generalize to higher-dimensional problems and constrained KKT systems (next section of the tutorial).
"""

# ╔═╡ 26c887ce-d95b-4e38-9717-e119de0e80ca
md"""
## Part III — Constrained Optimization (KKT)

We now move from unconstrained to **equality-constrained** minimization, then note what changes for **inequalities**.

##### 1) Problem and picture

We solve

```math
\min_{x\in\mathbb{R}^n} f(x)\quad\text{s.t.}\quad C(x)=0,\qquad C:\mathbb{R}^n\to\mathbb{R}^m.
```

**Geometry.** At an optimal $x^\star$ on the manifold $C(x)=0$, the gradient must be orthogonal to all feasible directions (the **tangent space**). Equivalently, $\nabla f(x^\star)$ lies in the span of the constraint normals:

```math
\nabla f(x^\star) + J_C(x^\star)^{\!T}\lambda^\star = 0, \qquad C(x^\star)=0.
```

This is the KKT stationarity + feasibility for equalities. (Under a mild **constraint qualification** like LICQ, these conditions are necessary at a local minimizer.)

**Lagrangian.** $L(x,\lambda)=f(x)+\lambda^{T}C(x)$. 

KKT (equalities only): $\nabla_x L(x,\lambda)=0,\ C(x)=0$. 
"""
 

# ╔═╡ 52718e0b-f958-445e-b9b6-9e5baf09e81a
md""" 
##### 2) Where the KKT linear system comes from 

At iterate $(x,\lambda)$, take a Newton step $(\Delta x,\Delta\lambda)$ by **linearizing** the two KKT equations:

* **Feasibility** $C(x+\Delta x)\approx C(x)+J_C(x)\,\Delta x=0 \;\Rightarrow\; J_C(x)\,\Delta x=-C(x)$.
* **Stationarity** $\nabla_x L(x+\Delta x,\lambda+\Delta\lambda)\approx \nabla_x L(x,\lambda)+H\,\Delta x+J_C(x)^{T}\Delta\lambda=0$,

where $H=\nabla^2_{xx}L(x,\lambda)$.

with the (approximate) **Lagrangian Hessian**

```math
H \;\approx\; \nabla^2 f(x)\;+\;\sum_{i=1}^m \lambda_i\,\nabla^2 C_i(x).
```

Stacking these two linearized equations gives the **saddle-point (KKT) system**:

```math
\begin{bmatrix}
H & J_C^{\!T}\\
J_C & 0
\end{bmatrix}
\begin{bmatrix}\Delta x\\ \Delta\lambda\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(x)+J_C^{\!T}\lambda\\
C(x)
\end{bmatrix}.
```

This is exactly the equality-constrained Newton step (feasible or infeasible start variants differ only in the right-hand side).

**Two common (H) choices.**

* **Full Newton (Lagrangian Hessian):** as above — best local rate near a solution.
* **Gauss–Newton/SQP:** drop the constraint-curvature term, taking ($H\approx\nabla^2 f(x)$) which is often more robust far from the solution.

**How we actually solve it.** The matrix is symmetric indefinite. Standard tactics:

* **Block elimination (Schur complement):** if ($H$) is nonsingular,

```math
(J_C H^{-1}J_C^{\!T})\,\Delta\lambda \;=\; J_C H^{-1}(\nabla f+J_C^{\!T}\lambda)+C,
\quad\text{then back-solve for }\Delta x.
```

* **LDLᵀ (symmetric-indefinite) factorization** for sparse problems.
"""

# ╔═╡ edce5b27-9af8-4010-9d9f-60681b2f427c
md"""
##### 3) Inequalities: what changes (just the essentials)

For $c(x)\ge 0$ (component-wise), first-order KKT add **sign** and **complementarity**:

```math
\begin{aligned}
&\text{Stationarity:}& &\nabla f(x)-J_c(x)^{\!T}\lambda=0,\\
&\text{Primal feasibility:}& &c(x)\ge 0,\\
&\text{Dual feasibility:}& &\lambda\ge 0,\\
&\text{Complementarity:}& &\lambda^{\!T}c(x)=0\quad(\lambda_i\,c_i(x)=0).
\end{aligned}
```

**Interpretation.** If constraint $i$ is **active** $(c_i(x)=0)$, its multiplier may be positive; if **inactive** $(c_i(x)>0)$, then $\lambda_i=0$.

"""

# ╔═╡ 4a3bdacd-bc17-4b10-bbed-6d34f0531d60
md"""
##### Small example

We will now do a tiny code example of equality-constrained problem and solve it with a Newton/SQP-style **KKT step** plus a simple **merit-function line search**. The goal is to see the geometry and the update.

We solve a **single equality constraint** problem in $\mathbb{R}^2$:
```math
\min_x\; f(x)\qquad \text{s.t. } c(x)=0.
```

We’ll use a convex quadratic objective and a curved constraint:

```math
f(x)=\tfrac12(x-x_\star)^\top Q(x-x_\star),\qquad
c(x)=x_1^2+2x_1-x_2.
```

A Newton/SQP step ($\Delta x,\Delta\lambda$) solves the **KKT system**

```math
\begin{bmatrix} H & J^\top \\ J & 0 \end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta \lambda \end{bmatrix}
=-
\begin{bmatrix} \nabla f(x)+J^\top\lambda \\ c(x) \end{bmatrix},
\quad J=\nabla c(x)^\top,\ \ H\approx\nabla^2\!L(x,\lambda).
```

We’ll compare two (H) choices:

* **Gauss–Newton/SQP:** ($H=\nabla^2 f$) (drop constraint curvature).
* **Full:** ($H=\nabla^2 f + \lambda,\nabla^2 c$).

To globalize we use the **merit** ($\phi(x)=f(x)+\tfrac{\rho}{2}c(x)^2$ with Armijo backtracking.
"""

# ╔═╡ 20fa9f12-ef19-482a-be83-feaed95109c3
begin
	# objective: bowl centered at x⋆
	Q_kkt     = Diagonal([0.5, 1.0])
	xstar_kkt = [1.0, 0.0]

	f_kkt(x::AbstractVector)  = 0.5 * dot(x .- xstar_kkt, Q_kkt * (x .- xstar_kkt))
	∇f_kkt(x::AbstractVector) = Q_kkt * (x .- xstar_kkt)
	∇²f_kkt(::AbstractVector) = Q_kkt  # constant SPD

	# one equality constraint: x₂ = x₁² + 2x₁
	c_kkt(x::AbstractVector)  = x[1]^2 + 2x[1] - x[2]
	∇c_kkt(x::AbstractVector) = [2x[1] + 2.0, -1.0]
	∇²c_kkt(::AbstractVector) = [2.0 0.0; 0.0 0.0]
end

# ╔═╡ 6c560afb-921d-4f34-b77b-a31e37b7d571
begin
	ϕ_kkt(x; ρ=1.0)  = f_kkt(x) + 0.5*ρ*c_kkt(x)^2
	∇ϕ_kkt(x; ρ=1.0) = ∇f_kkt(x) .+ ρ*c_kkt(x).*∇c_kkt(x)

	"""
	kkt_step(x, λ; method=:gn, δ=1e-8)

	Build and solve the 3×3 KKT system:
	  [ H  J';  J  0 ] [Δx; Δλ] = -[ ∇f + J'λ;  c ]
	with H = ∇²f (method=:gn) or H = ∇²f + λ∇²c (method=:full).
	δ adds a tiny ridge to H for numerical robustness.
	"""
	function kkt_step(x::AbstractVector, λ::Real; method::Symbol=:gn, δ::Float64=1e-8)
		H = Matrix(∇²f_kkt(x))
		if method === :full
			H .+= λ .* ∇²c_kkt(x)
		end
		@inbounds for i in 1:2; H[i,i] += δ; end
		J = reshape(∇c_kkt(x), 1, :)             # 1×2
		rhs = -vcat(∇f_kkt(x) .+ λ .* ∇c_kkt(x), c_kkt(x))
		K   = [H J'; J 0.0]
		Δ   = try
			K \ rhs
		catch
			pinv(K) * rhs
		end
		return Δ[1:2], Δ[3]
	end

	"""
	kkt_solve(x0, λ0; iters=8, method=:gn, linesearch=true, ρ=1.0, b=1e-4, cdec=0.5)

	A few KKT iterations with Armijo on ϕ. Returns (X, Λ) where
	X ∈ ℝ^{2×(iters+1)} collects iterates.
	"""
	function kkt_solve(x0, λ0; iters=8, method=:gn, linesearch=true, ρ=1.0, b=1e-4, cdec=0.5)
		x = copy(x0); λ = float(λ0)
		X = reshape(x, 2, 1); Λ = [λ]
		for _ in 1:iters
			Δx, Δλ = kkt_step(x, λ; method=method)

			# Armijo w/ descent fallback on φ
			α = 1.0
			if linesearch
				φx   = ϕ_kkt(x; ρ=ρ)
				gφ   = ∇ϕ_kkt(x; ρ=ρ)
				slope = dot(gφ, Δx)
				if !(isfinite(slope)) || slope ≥ 0
					Δx .= -gφ
					Δλ  = 0.0
					slope = -dot(gφ, gφ)
				end
				for _ in 1:20
					if ϕ_kkt(x .+ α .* Δx; ρ=ρ) ≤ φx + b*α*slope
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

# ╔═╡ 4d8533ef-be1b-45c9-acaf-ce278c3e5db7
begin
    # Find constrained minimizer of f along the curve x₂ = x₁² + 2x₁ (within the window)
    function constrained_star_kkt(; xlim=(-4,4), ylim=(-4,4))
        xs = range(xlim[1], xlim[2], length=1201)
        ys = @. xs^2 + 2xs
        mask = (ys .>= ylim[1]) .& (ys .<= ylim[2])
        xs_in = collect(xs)[mask]; ys_in = ys[mask]
        if isempty(xs_in)      # fallback if curve is off-screen
            xs_in = collect(xs); ys_in = ys
        end
        vals = [ f_kkt([x,y]) for (x,y) in zip(xs_in, ys_in) ]
        i = argmin(vals)
        return xs_in[i], ys_in[i]
    end

    function landscape_plot_kkt(; xlim=(-4,4), ylim=(-4,4), nsamp=121)
        xs = range(xlim[1], xlim[2], length=nsamp)
        ys = range(ylim[1], ylim[2], length=nsamp)
        Z  = [ f_kkt([x,y]) for y in ys, x in xs ]

        p = plt.contour(xs, ys, Z;
            levels=18, xlabel="x₁", ylabel="x₂",
            title="KKT path on objective contours (c(x)=0)",
            legend=:bottomright, aspect_ratio=:equal,
            xlims=(xlim[1], xlim[2]), ylims=(ylim[1], ylim[2]))

        # constraint curve
        xc = collect(xs); yc = @. xc^2 + 2xc
        plt.plot!(p, xc, yc; lw=2, label="c(x)=0")
 

        xcs, ycs = constrained_star_kkt(xlim=xlim, ylim=ylim)
        plt.scatter!(p, [xcs], [ycs];
                     marker=:star5, ms=9, label="constrained x̂")

        return p
    end

   # Path overlay: X's for iterates, star for the current x0 from sliders
	plot_path_kkt!(p, X; x0::AbstractVector = X[:,1]) = begin
	    # iterates
	    plt.plot!(p, X[1,:], X[2,:]; marker=:x, ms=6, lw=1.5, label="iterates") 
	    plt.scatter!(p, [x0[1]], [x0[2]];  marker=:star5, ms=9, label="start x₀")
	    plt.xlims!(p, (-4,4)); plt.ylims!(p, (-4,4))
	    p
	end

end


# ╔═╡ 79604603-df5e-47de-aeb8-08e7658fe190
md"""
### Controls

**Start**  
x₁ $(@bind x1_kkt Slider(-3.5:0.1:3.5, default=-1.5, show_value=true))  
x₂ $(@bind x2_kkt Slider(-3.0:0.1:3.0, default=-1.0, show_value=true))

**Dual init**  
λ₀ $(@bind λ0_kkt Slider(-2.0:0.1:2.0, default=0.0, show_value=true))

**Solver**  
Iters $(@bind iters_kkt Select([3,5,8,10,15,20]; default=8))  
Method $(@bind method_kkt Select([:gn, :full]; default=:gn))  
Line search $(@bind use_ls_kkt CheckBox(default=true))  
ρ (merit weight) $(@bind rho_kkt Slider(0.1:0.1:5.0, default=1.0, show_value=true))

$(@bind run_kkt Button("Run KKT"))
"""

# ╔═╡ c249309e-a47c-48d9-9b06-d448e0a57a28
let
    run_kkt
	
    x1_kkt; x2_kkt; λ0_kkt; iters_kkt; method_kkt; use_ls_kkt; rho_kkt

    x0 = [x1_kkt, x2_kkt]
    X, Λ = kkt_solve(x0, λ0_kkt; iters=iters_kkt, method=method_kkt,
                     linesearch=use_ls_kkt, ρ=rho_kkt)

    fig = landscape_plot_kkt()
    plot_path_kkt!(fig, X; x0=x0)  

    feas = abs(c_kkt(X[:,end]))
    stat = norm( ∇f_kkt(X[:,end]) .+ (Λ[end] .* ∇c_kkt(X[:,end])) ) 
    fig
end

# ╔═╡ 917ed8f6-d451-4725-a794-37b6bc7e46f5
md"""
**What to notice**

- The iterate stays close to the constraint curve (feasibility via the KKT step).
- **Gauss–Newton** ($H=\nabla^2 f$) is usually robust from farther starts.
- **Full** ($H=\nabla^2 f + \lambda\nabla^2 c$) accelerates near a solution.
- Armijo on $\phi(x)=f(x)+\tfrac{\rho}{2}c(x)^2$ avoids oscillation and keeps the path stable across slider settings.
"""

# ╔═╡ c0a75eef-e3e1-4e5f-8184-292827870cb2
md"""
# Part IV — Interior-Point Method

##### Why Interior-Point?

We want to solve an inequality-constrained problem

$\min_x\ f(x)\quad\text{s.t.}\quad c(x)\ge 0$

Interior-point methods keep the iterate **strictly inside** the feasible set and add a
**barrier** that explodes as we approach the boundary:

$F_\mu(x)\;=\;f(x)\;-\;\mu\,\log c(x),\qquad \mu>0$

For a fixed $\mu$ we take a few Newton steps on $F_\mu$ (with a feasibility-aware line
search). Then we **decrease** $\mu$ and repeat. As $\mu\downarrow 0$, the minimizer of $F_\mu$
slides along the **central path** toward the true constrained optimum.


More concretely, we solve the inequality problem by following minimizers of the **barrier objective**
$F_\mu(x) \;=\; f(x)\;-\;\mu\,\log c(x), \qquad \mu>0$,
staying strictly inside $c(x)>0$ and shrinking $\mu$.

**Inputs.**
- feasible start $x_0$ with $c(x_0)>0$  (or “interiorize” a guess),
- barrier $\mu_0>0$, shrink factor $\tau\in(0,1)$ (e.g. $\tau=0.3$),
- inner Newton tolerance / max steps.

**Derivatives used by Newton.**: The gradient and Hessian of the barrier objective are

$\nabla F_\mu(x)=\nabla f(x)\;-\;\frac{\mu}{c(x)}\,\nabla c(x)$
$\nabla^2 F_\mu(x)=\nabla^2 f(x)
\;+\;\frac{\mu}{c(x)^2}\,\nabla c(x)\nabla c(x)^\top
\;-\;\frac{\mu}{c(x)}\,\nabla^2 c(x)$

**Inner loop (Newton with feasibility-aware backtracking, fixed $\mu$).**

1. Compute the Newton direction by solving

$\nabla^2 F_\mu(x_k)\,\Delta x_k \;=\; -\,\nabla F_\mu(x_k)$

2. Choose step length $\alpha \in \{1,\; \beta,\; \beta^2,\ldots\}$ (e.g. $beta=0.5$) until both hold:
   - **interior:** $c(x_k+\alpha\Delta x_k) > 0$
   - **Armijo decrease:** 

     $F_\mu(x_k+\alpha\Delta x_k) \;\le\;
     F_\mu(x_k) + b\,\alpha\,\nabla F_\mu(x_k)^\top \Delta x_k$ with $b\in(0,1)$ (e.g. $b=10^{-4}$).
3. Update $x_{k+1} \leftarrow x_k+\alpha\Delta x_k$.
4. Stop the inner loop if $\|\nabla F_\mu(x_{k+1})\|$ is small or the step count is reached.

**Outer loop (reduce the barrier, warm-start).**

1. Set $\mu \leftarrow \tau\,\mu$.
2. Re-run the inner loop starting at the last $x$.
3. Stop when $\mu \le \mu_{\min}$ (or changes in $x$ become tiny).

This produces a sequence that tracks the **central path** and approaches the constrained minimizer as $\mu\downarrow 0$. 
"""


# ╔═╡ 60cc80f1-818b-45d4-8248-bf2e0bfb6936
md"""
##### Our tiny example 

Objective (a shifted quadratic bowl):

$f(x)=\tfrac12\,(x-x_\star)^\top Q\,(x-x_\star),
\qquad
Q=\mathrm{diag}(0.5,\,1),\quad x_\star=\begin{bmatrix}1 \\ 0\end{bmatrix}$


Curved inequality (feasible region **above** the curve):

$c(x)=x_2-\bigl(x_1^2+2x_1\bigr)\ \ge 0$

The unconstrained minimizer $x_\star=(1,0)$ is **infeasible** for this constraint, so the true solution lies **on the boundary** $c(x)=0$.
In the plot you’ll see:
- the boundary $x_2=x_1^2+2x_1$ (orange),
- a gold star at $x_\star$ (for reference),
- red **×** markers for the **barrier Newton iterates**.

Two practical details:

1. We keep iterates strictly feasible with a tiny **interiorize** step if $c(x)\le 0$.  
2. The line search accepts steps only if $c(x+\alpha\Delta x)>0$ **and**
   $F_\mu$ decreases (Armijo condition).
"""

# ╔═╡ 8a5029b2-2355-4b9b-84d5-50970c6a4b44
begin 

    # Quadratic bowl (unconstrained minimizer at x⋆ = (1, 0))
    Q      = Diagonal([0.5, 1.0])
    xstar  = [1.0, 0.0]
    f(x)   = 0.5 * dot(x .- xstar, Q * (x .- xstar))
    ∇f(x)  = Q * (x .- xstar)
    ∇²f(::AbstractVector) = Q

    # Curved inequality: c(x) = x1^2 + 2 x1 - x2  ≥ 0
    c(x)    = x[2] - (x[1]^2 + 2x[1])
	∇c(x)   = [-2x[1] - 2.0,  1.0]
	∇²c(::AbstractVector) = [-2.0  0.0;  0.0  0.0]   # constant

    # Barrier objective pieces
    Fμ(x, μ)   = f(x) - μ * log(c(x))
    ∇Fμ(x, μ)  = ∇f(x) .- (μ / c(x)) .* ∇c(x)
    ∇²Fμ(x, μ) = ∇²f(x) .+ (μ / (c(x)^2)) .* (∇c(x) * ∇c(x)') .- (μ / c(x)) .* ∇²c(x)

    # Keep the iterate strictly feasible: nudge along +∇c until c(x) > eps
    function interiorize(x; eps=1e-3)
        cx = c(x)
        cx > eps && return x
        g  = ∇c(x)
        t  = (eps - cx) / max(norm(g)^2, 1e-12)   # first-order step
        return x .+ t .* g
    end

    # A few Newton steps on Fμ with Armijo + feasibility guard
    function barrier_newton(x0, μ; steps=8, b=1e-4, shrink=0.5)
        x = interiorize(x0)
        X = reshape(x, 2, 1)
        for _ in 1:steps
            g = ∇Fμ(x, μ)
            H = ∇²Fμ(x, μ)                 # SPD near the interior
            Δ = - H \ g
            # Backtrack until c(x+αΔ)>0 and Fμ decreases
            α = 1.0
            Fx = Fμ(x, μ); slope = dot(g, Δ)
            while true
                x_try = x .+ α .* Δ
                if c(x_try) > 0 && Fμ(x_try, μ) <= Fx + b*α*slope
                    x = x_try
                    break
                end
                α *= shrink
                if α < 1e-12; break; end
            end
            X = hcat(X, x)
            if norm(g) < 1e-10; break; end
        end
        return X
    end

    # Plot: contours, constraint, stars, iterates
    function ipm_plot(X; legend=:bottomright)
        xs = range(-4, 4, length=181)
        ys = range(-4, 4, length=181)
        Z = [ f([x,y]) for y in ys, x in xs ]
        p  = plt.contour(xs, ys, Z; levels=18, aspect_ratio=:equal,
                         xlims=(-4,4), ylims=(-4,4),
                         xlabel="x₁", ylabel="x₂",
                         title="Interior-Point (log-barrier)",
                         legend=legend)
        # constraint curve: x2 = x1^2 + 2 x1
        xc = collect(xs); yc = @. xc^2 + 2xc
        plt.plot!(p, xc, yc; lw=2, label="c(x)=0 (boundary)")
        # stars
        plt.scatter!(p, [xstar[1]], [xstar[2]]; marker=:star5, ms=9, label="unconstrained x⋆")
        # start + iterates
        plt.scatter!(p, [X[1,1]], [X[2,1]]; marker=:star5, ms=9, label="start x₀")
        plt.plot!(p, vec(X[1,:]), vec(X[2,:]); marker=:x, ms=6, lw=2, label="barrier iterates")
        return p
    end
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


# ╔═╡ 8e66acef-684d-4536-ae9b-49ce6e8dc24b
md"""
##### How to read the figure

- Pick a start \(x_1,x_2\) and a barrier strength $\mu=10^k$ with the sliders.
- Click **Run IPM**: the code takes a handful of Newton steps on $F_\mu$.
- Large $\mu$: iterates stay well inside; Small $\mu$: iterates **hug the boundary**
  and move toward the constrained minimum.

**Sanity checks as you play:**
- Check $c(x)$ stays **positive** along the path (interior).  
- Check $F_\mu$ is **monotone decreasing** across accepted steps.  
- As you reduce $\mu$, the final point gets closer to the boundary optimum.
"""

# ╔═╡ 2ff89961-af3e-4792-8f71-3f2a2ca53056
md"""
### IPM controls

Start x₁  $(@bind x1 PlutoUI.Slider(-3.0:0.1:2.0, default=-2.0, show_value=true))  
Start x₂  $(@bind x2 PlutoUI.Slider(-2.0:0.1:4.0, default= 2.0, show_value=true))

Barrier exponent \(k\) (we use \(\mu=10^k\))  
k = $(@bind k PlutoUI.Slider(-6:0.05:0, default=-3, show_value=true))

Steps $(@bind steps PlutoUI.Select([4,6,8,10,15,20,25,50,100]; default=8))

$(@bind run_p4 PlutoUI.Button("Run IPM"))
"""


# ╔═╡ eb4d1779-2283-44ca-addb-47713e04948d
let
    run_p4
    μ = 10.0^k
    X = barrier_newton([x1, x2], μ; steps=steps)
    ipm_plot(X)
end


# ╔═╡ e06d3761-77a6-4e84-a3ea-d13b6d0c57dd
md""" 

The takeaway from IPM in a nutshell is that IPM consists of repeating the following steps in iterative fashion:


$\textbf{repeat:}\quad
\begin{cases}
\text{(inner) solve } \min_x F_\mu(x)\ \text{by Newton + backtracking},\\[2pt]
\text{(outer) } \mu \leftarrow \tau\,\mu\quad(0<\tau<1)\ \text{and warm-start}.
\end{cases}$

At a constrained solution $x^\star$, the KKT conditions read

$\nabla f(x^\star) - J_c(x^\star)^\top \lambda^\star = 0,\quad
c(x^\star)\ge 0,\quad \lambda^\star\ge 0,\quad
\lambda^{\star\top}c(x^\star)=0$
Barrier methods implicitly track $\lambda \approx \mu/c(x)$ and drive
$\lambda\circ c(x)\to 0$ as $\mu\downarrow 0$ (complementarity).
"""

# ╔═╡ 3132cda5-ad6e-4ae2-afba-ea18d09cf8df
md"""
# Part V - Sequential Quadratic Programming (SQP)

**Idea.** Solve a nonlinear constrained problem by repeatedly solving a **quadratic program (QP)** built from local models.

- Quadratic model of the Lagrangian/objective near the current iterate.
- Linearize the constraints around the current iterate.
- Each iteration: solve a QP to get a step $d$, then update $x \leftarrow x + \alpha d$.
- Strengths: Newton-like local convergence (often superlinear), warm-start friendly.

##### Problem & KKT recap

We consider

```math
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.}\quad g(x)=0,\ \ h(x)\le 0.
```

At a candidate optimum $x^\star$, the KKT conditions require multipliers $\lambda \in \mathbb{R}^m$, $\mu \in \mathbb{R}^p_{\ge 0}$ such that

```math
\nabla f(x^\star) + \nabla g(x^\star)^\top \lambda + \nabla h(x^\star)^\top \mu = 0
```

```math
g(x^\star)=0,\ \ h(x^\star)\le 0,\ \ \mu \ge 0,\ \ \mu \odot h(x^\star)=0
```

##### Local models that define the QP

At iterate $x_k$ with multipliers $(\lambda_k,\mu_k)$:

**Quadratic model**

```math
m_k(d) = \nabla f(x_k)^\top d + \tfrac{1}{2}\, d^\top B_k\, d
```

with $B_k \approx \nabla^2_{xx}\mathcal{L}(x_k,\lambda_k,\mu_k)$. Common choices: exact Hessian, (L-)BFGS, or Gauss–Newton.

**Linearized constraints**

```math
\begin{aligned}
g(x_k) + \nabla g(x_k)\, d &= 0,\\
h(x_k) + \nabla h(x_k)\, d &\le 0.
\end{aligned}
```

##### The SQP subproblem (QP)

```math
\begin{aligned}
\min_{d \in \mathbb{R}^n}\quad &
\nabla f(x_k)^\top d + \tfrac{1}{2}\, d^\top B_k\, d \\
\text{s.t.}\quad &
\nabla g(x_k)\, d + g(x_k) = 0, \\
&
\nabla h(x_k)\, d + h(x_k) \le 0.
\end{aligned}
```

Solving this QP yields a step $(d_k)$ and QP multipliers $(\lambda_{k+1},\mu_{k+1})$.
Update with $(x_{k+1} = x_k + \alpha_k d_k)$ (line search or trust region).

##### SQP (line-search flavor): 6 steps

1. Initialize $x_0$, multipliers $(\lambda_0,\mu_0)$, and $B_0 \succ 0$.
2. Build the QP at $x_k$ using $B_k$, $\nabla g(x_k)$, $\nabla h(x_k)$, $g(x_k)$, $h(x_k)$.
3. Solve the QP $\Rightarrow$ get $d_k$ and $(\lambda_{k+1},\mu_{k+1})$.
4. Choose $\alpha_k \in (0,1]$ via globalization (merit or filter).
5. Set $x_{k+1} = x_k + \alpha_k d_k$.
6. Update $B_{k+1}$ (e.g., damped BFGS). Stop when KKT residuals are small.

##### Globalization: make SQP robust

**Merit (penalty) function** for line search, e.g.
$\phi(x) \;=\; f(x) \;+\; \tfrac{\rho}{2}\,\|g(x)\|^2 \;+\; \rho\,\|h(x)_+\|_1,$
and choose $\alpha_k$ by Armijo/backtracking so $\phi$ decreases.

**Filter methods** accept steps that improve *either* objective *or* feasibility.

**Trust-region SQP**: restrict $\|d\|\le \Delta_k$, compare predicted vs actual reduction, adjust $\Delta_k$.

##### Inequalities & active sets (intuition)

- The QP contains the **linearized inequalities** $h(x_k)+\nabla h(x_k)d \le 0$.
- Its KKT system enforces complementarity via multiplier signs and active constraints.
- A small **working set** (estimated active constraints) tends to stabilize across iterations, enabling warm starts and fast solves.
"""

# ╔═╡ Cell order:
# ╟─81ebc291-89f0-4c1e-ac34-d5715977dd86
# ╟─9543f7bc-ab36-46ff-b471-9aa3db9739e4
# ╟─8969e78a-29b0-46d3-b6ba-59980208fe5b
# ╟─d90e9be0-7b68-4139-b185-6cbaad0d307e
# ╟─7b896268-4336-47e2-a8b5-f985bfde51f5
# ╟─342decc1-43fa-432a-9a9c-757a10ba6a5d
# ╟─fd1ad74b-cb74-49a7-80b8-1a282abfdff2
# ╟─5e6c4ea7-b283-423c-b9c0-421d53cebc2d
# ╠═49d5b2e6-eb29-478c-b817-8405d55170b1
# ╟─950c61b8-f076-4b9a-8970-e5c2841d75f2
# ╟─80035b9c-eba6-469c-b138-c6c792979493
# ╠═92841a2e-bc0d-40f8-8344-a5c398a67275
# ╟─8813982c-8c9a-4706-91a8-ebadf9323a4f
# ╠═4307a2f3-0378-4282-815b-9ed1fa532e1c
# ╟─a45ed97f-f7c1-4ef5-9bc7-654e827f751b
# ╠═5a17f83e-751b-4244-9c15-7165645bfe29
# ╟─56c965c9-5acc-40a5-b1dd-c3a59f0462a9
# ╟─af82d16d-c649-461b-856a-42355517d9f4
# ╟─662d58d7-4c9c-4699-a846-cb6070c507d9
# ╠═27760607-b740-47dc-a810-c332baa2bd2d
# ╠═a098a49b-a368-4929-824c-385a06b88696
# ╟─17ac372e-87e6-4649-ba9d-1df1cdb7b55b
# ╟─a62f1b6a-87fe-4401-bc13-42166ca0e129
# ╟─be41026f-cb28-4647-8db6-1d243739f444
# ╠═b2ddc048-4942-43bc-8faa-1921062d8c9c
# ╟─1ffa2941-619b-400a-ba0f-56baa6ee7f59
# ╠═6fd7a753-6db7-4c37-9e22-dc63dd3072c8
# ╠═49e0f5c3-fe14-42e1-9b3a-83c4447148a8
# ╠═096ebc95-f133-4ca3-b942-cf735faaa42b
# ╟─fdf41c76-4405-49e0-abfa-5c5193de99f4
# ╠═22bfe0a3-c61b-4dfe-8f20-a8bf807c2e14
# ╟─d259c1b8-3716-4f80-b462-3b3daebb444d
# ╟─76c51e32-bd0e-4e72-8c47-64352da13d3e
# ╟─858d4ee1-4f15-4ced-b984-c0291237d359
# ╟─e012d7e0-0181-49d0-bb78-995378c4f87a
# ╟─26c887ce-d95b-4e38-9717-e119de0e80ca
# ╟─52718e0b-f958-445e-b9b6-9e5baf09e81a
# ╟─edce5b27-9af8-4010-9d9f-60681b2f427c
# ╟─4a3bdacd-bc17-4b10-bbed-6d34f0531d60
# ╠═20fa9f12-ef19-482a-be83-feaed95109c3
# ╠═6c560afb-921d-4f34-b77b-a31e37b7d571
# ╠═4d8533ef-be1b-45c9-acaf-ce278c3e5db7
# ╟─79604603-df5e-47de-aeb8-08e7658fe190
# ╠═c249309e-a47c-48d9-9b06-d448e0a57a28
# ╟─917ed8f6-d451-4725-a794-37b6bc7e46f5
# ╟─c0a75eef-e3e1-4e5f-8184-292827870cb2
# ╟─60cc80f1-818b-45d4-8248-bf2e0bfb6936
# ╠═8a5029b2-2355-4b9b-84d5-50970c6a4b44
# ╟─8e66acef-684d-4536-ae9b-49ce6e8dc24b
# ╟─2ff89961-af3e-4792-8f71-3f2a2ca53056
# ╠═eb4d1779-2283-44ca-addb-47713e04948d
# ╟─e06d3761-77a6-4e84-a3ea-d13b6d0c57dd
# ╟─3132cda5-ad6e-4ae2-afba-ea18d09cf8df
