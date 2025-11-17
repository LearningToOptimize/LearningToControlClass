### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ f65b22ca-183c-4cd2-a9af-0b47e90df929
begin
    import Pkg
    Pkg.add("PlutoUI")
    Pkg.add("PlutoTeachingTools")
    Pkg.add("MarkdownLiteral")
    Pkg.add("DifferentialEquations")      
end

# ╔═╡ ed800160-c3ed-11f0-0aff-9b9e6e4a3d5a
begin
	class_dir = @__DIR__
	
	Pkg.activate(".")
	Pkg.instantiate()
	
	using PlutoUI
	using PlutoTeachingTools
	using MarkdownLiteral
end

# ╔═╡ 30fe161b-83af-4199-bf90-6a188d632738
begin
    Pkg.add.(["DifferentialEquations","NLsolve","JuMP","Ipopt","CairoMakie","PlutoTeachingTools"])
end

# ╔═╡ 8b5bf689-a835-49df-82d3-f0de8ed0d444
begin
    using LinearAlgebra
    using DifferentialEquations
    using NLsolve
    using JuMP
    using Ipopt
    using CairoMakie
end

# ╔═╡ 1c00dbc5-36d6-4fdd-bb8b-2c171a934653
ChooseDisplayMode()

# ╔═╡ af09aa0e-88bd-4f6d-93b3-c35144133fb5
md"
| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Zaowei Dai |
|  Date   | : | 5 of September, 2025 |
"

# ╔═╡ 8b31afac-acad-46df-a213-6921567cce04
md"""
# Lecture: Pontryagin's Maximum Principle, Shooting Methods, and LQR

In this notebook we will:

1. Introduce **Pontryagin's Maximum Principle (PMP)** as a necessary condition for optimal control.
2. Show how the resulting **two–point boundary value problem (BVP)** can be solved using **shooting** and **multiple shooting**.
3. Specialize to **Linear-Quadratic Regulator (LQR)** problems, derive the **Riccati equation**, and show the **QP viewpoint**.
"""

# ╔═╡ 55c540b6-f463-4628-b5f8-5b835d15c3c7


# ╔═╡ 8422c150-e56c-4e60-a8ed-9aef1af39339
md"
# Pontryagin's Maximum Principle

We consider the finite-horizon optimal control problem:

```math
\begin{align*}
\min_{u(\cdot)} \quad & J(x,u)
    = \Phi(x(T)) + \int_0^T \ell(x(t),u(t),t)\,dt \\
\text{s.t.} \quad
& \dot{x}(t) = f(x(t),u(t),t), \quad t\in[0,T], \\
& x(0) = x_0, \\
& u(t) \in U \subset \mathbb{R}^m .
\end{align*}
```

We define the Hamiltonian

```math
H(x,u,\lambda,t)
  := \ell(x,u,t) + \lambda^\top f(x,u,t).
```

Pontryagin's Maximum Principle [^liberzon] states that an optimal trajectory
\((x^\*,u^\*,\lambda^\*)\) must satisfy:

```math
\begin{align*}
\dot x^\*(t) &= \nabla_\lambda H(x^\*,u^\*,\lambda^\*,t), \\
\dot \lambda^\*(t) &= -\nabla_x H(x^\*,u^\*,\lambda^\*,t), \\
\lambda^\*(T) &= \nabla_x \Phi(x^\*(T)), \\
u^\*(t) &\in \arg\min_{u\in U} H(x^\*(t),u,\lambda^\*(t),t).
\end{align*}
```

This is a two-point boundary value problem (BVP):

- the state has an initial condition \(x(0)=x_0\);
- the costate has a terminal condition \(\lambda(T)=\nabla_x\Phi(x(T))\).
"


# ╔═╡ 647a107a-1fb1-4a30-8ea5-57fbcdec96d3
question_box(md"""
**Q1. Why does the terminal condition say that the optimal costate at the final time equals the gradient of the terminal cost with respect to the state?**
""")

# ╔═╡ c47669aa-71c0-49b0-9185-7eacf79ebb25
Foldable(md"Q1 – Discussion",
md"""
In Pontryagin’s Maximum Principle, we look at how a small change in the control produces a small change in the state trajectory and in the final state. This in turn produces a first–order change in the performance index, which has two parts: the running cost along the trajectory and the terminal cost evaluated at the final state.

When we derive the optimality conditions, we introduce the costate to rewrite the variation of the running cost and then integrate by parts. By choosing the costate dynamics appropriately, we cancel all interior terms that involve variations of the state inside the time interval. After this cancellation, the only remaining term that still contains the state variation is a boundary term at the final time. This boundary term involves the difference between the costate at the final time and the gradient of the terminal cost with respect to the state, multiplied by the variation of the final state.

For an optimal trajectory, the first–order variation of the performance index must be zero for any admissible small change in the control, and therefore for any induced small change in the final state. The only way for the boundary term to always vanish is that the factor multiplying the variation of the final state is identically zero. This means that the costate at the final time must be exactly equal to the gradient of the terminal cost with respect to the state evaluated at the optimal final state. 
""")

# ╔═╡ 8fda68ae-6a02-4890-945a-5054c51c53e4
md"
## Scalar Linear-Quadratic Example

Consider the scalar system

```math
\begin{align*}
\dot x(t) &= a x(t) + b u(t), \\
x(0) &= x_0 .
\end{align*}
```

with cost

```math
\begin{align*}
J = \frac12 p_f x(T)^2
+ \int_0^T \left(
    \frac12 q x(t)^2
  + \frac12 r u(t)^2
\right)\,dt .
\end{align*}
```

The Hamiltonian is

```math
H(x,u,\lambda)
  = \frac12 q x^2 + \frac12 r u^2 + \lambda (a x + b u).
```

The costate equation is

```math
\dot\lambda(t) = -\frac{\partial H}{\partial x}
               = -q x(t) - a \lambda(t).
```

Stationarity in \(u\) gives

```math
\frac{\partial H}{\partial u}
  = r u + b \lambda = 0
  \quad\Rightarrow\quad
u^\*_{\text{uncon}}(t) = -\frac{b}{r} \lambda(t).
```

With a control bound \(|u|\le u_{\max}\),

```math
u^\*(t)
  = \mathrm{sat}_{[-u_{\max},u_{\max}]}
    \left(-\frac{b}{r}\lambda(t)\right).
```

The boundary conditions are

```math
x(0) = x_0,
\qquad
\lambda(T) = p_f x(T).
```
"

# ╔═╡ a2566189-e807-4a08-ae0a-56fbe92a19c9
begin
    # Parameters for the scalar LQ example
    a   = -1.0      # system drift
    b   = 1.0       # input gain
    q   = 1.0       # state weight
    r   = 0.1       # control weight
    p_f = 5.0       # terminal weight
    u_max = 1.0     # control bound
    T    = 5.0      # horizon
    x0   = 1.0      # initial state

    sat(u, umax) = clamp(u, -umax, umax)

    # Joint PMP dynamics: z = [x; λ]
    function pmp_dynamics!(dz, z, p, t)
        x, λ = z
        u = sat(-b / r * λ, u_max)
        dx = a * x + b * u
        dλ = -q * x - a * λ
        dz[1] = dx
        dz[2] = dλ
    end
end


# ╔═╡ 0659e29f-c6e0-4881-99b9-25a21214af37
md"
## Shooting Method for the PMP BVP

We treat the unknown initial costate as a scalar parameter \(s = \lambda(0)\).
For any fixed \(s\), we integrate

```math
\dot z(t) =
\begin{bmatrix}
\dot x(t) \\[2pt]
\dot\lambda(t)
\end{bmatrix},
\quad
z(0) =
\begin{bmatrix}
x_0 \\ s
\end{bmatrix},
\quad
t\in[0,T].
```

At time \(T\) we obtain \(x(T;s)\) and \(\lambda(T;s)\) and define the residual

```math
F(s) = \lambda(T; s) - p_f x(T; s).
```

The shooting method solves the scalar equation

```math
F(s^\*) = 0,
```

and then uses \(s^\* = \lambda^\*(0)\) to define the optimal state, costate,
and control trajectories.
"


# ╔═╡ 202312d6-89c6-430b-ad21-63397c7f57b7
question_box(md"""
**Q2. What can go wrong with single shooting for long horizons or unstable dynamics?**
""")

# ╔═╡ 5a80124b-8671-4262-9178-cb1696d244ca
Foldable(md"Q2 – Discussion",
md"""
In single shooting we treat the unknown initial costate as a parameter and define the residual

\[
F(s) = \lambda(T; s) - p_f x(T; s).
\]

We then solve \(F(s) = 0\) by a root-finding method.

For unstable dynamics or large horizons \(T\),

- small changes in the initial guess \(s\) can produce **exponentially large changes**
  in \(x(T;s)\) and \(\lambda(T;s)\);
- the derivative \(F'(s)\) can be extremely large or extremely small
  (ill-conditioned scalar problem);
- numerical integration errors accumulate over the whole interval \([0,T]\).

As a result, Newton-type methods may diverge or converge very slowly.
This motivates more robust approaches such as **multiple shooting** and **direct collocation**.
""")


# ╔═╡ 780a258c-8e2d-4822-ae0a-dc2454574009
begin
    # Integrate PMP dynamics forward for a given initial costate s = λ(0)
    function simulate_pmp(s; T=T)
        z0 = [x0, s]
        prob = ODEProblem(pmp_dynamics!, z0, (0.0, T))
        sol = solve(prob, Tsit5(); reltol=1e-8, abstol=1e-8)
        return sol
    end

    # Scalar shooting residual F(s)
    function shooting_residual(s)
        sol = simulate_pmp(s; T=T)
        xT = sol(T)[1]
        λT = sol(T)[2]
        return λT - p_f * xT
    end

    # NLsolve wrapper: F_vec[1] = F(s_vec[1])
    function shooting_equation!(F_vec, s_vec)
        s = s_vec[1]
        F_vec[1] = shooting_residual(s)
    end

    # Initial guess for λ(0)
    s0 = [0.0]
    shoot_sol = nlsolve(shooting_equation!, s0)
    λ0_star = shoot_sol.zero[1]

    @info "Optimal initial costate λ(0)" λ0_star
end


# ╔═╡ c6ffb973-b201-47f5-88c3-d52ad5dda5cd
begin
    # Simulate and plot optimal trajectories for the scalar example
    sol_star = simulate_pmp(λ0_star; T=T)

    ts = range(0, T; length=400)
    xs = [sol_star(t)[1] for t in ts]
    λs = [sol_star(t)[2] for t in ts]
    us = [sat(-b / r * λ, u_max) for λ in λs]

    fig = Figure(resolution = (900, 600))

    ax1 = Axis(fig[1, 1],
               title = "State x(t) and costate λ(t) (single shooting)",
               xlabel = "t", ylabel = "value")
    lines!(ax1, ts, xs, label = "x(t)")
    lines!(ax1, ts, λs, label = "λ(t)")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[2, 1],
               title = "Optimal control u(t)",
               xlabel = "t", ylabel = "u(t)")
    lines!(ax2, ts, us, label = "u(t)")
    axislegend(ax2, position = :rt)

    fig
end


# ╔═╡ be0ab297-eeee-471b-96a0-3fa7301be244
md"
# LQR, Riccati, and QP Viewpoint

## Finite-horizon discrete-time LQR

We consider the linear system

```math
x_{k+1} = A x_k + B u_k, \quad k = 0,\dots,N-1,
```

with quadratic cost

```math
J = \frac12 x_N^\top Q_f x_N
  + \sum_{k=0}^{N-1}
    \left(
      \frac12 x_k^\top Q x_k
      + \frac12 u_k^\top R u_k
    \right),
```

where \(Q,Q_f \succeq 0\) and \(R \succ 0\).

The optimal control has the form

```math
u_k^\* = -K_k x_k,
```

where the feedback gains \(K_k\) come from the backward Riccati recursion [^anderson].

Let \(P_N = Q_f\). For \(k = N-1,\dots,0\),

```math
\begin{align*}
K_k &= (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A, \\
P_k &= Q + A^\top P_{k+1} A
      - A^\top P_{k+1} B
        (R + B^\top P_{k+1} B)^{-1}
        B^\top P_{k+1} A.
\end{align*}
```
"


# ╔═╡ 95cfd865-6c84-46a9-8ef7-2c2cc241740d
begin
    A = [1.0 1.0;
         0.0 1.0]

    # Make B a 2×1 matrix, NOT a vector
    B = [0.0 1.0]'        # 2×1 matrix

    Q  = Diagonal([1.0, 0.1])
    R  = Diagonal([0.01])  # keep R as a 1×1 diagonal matrix
    Qf = Diagonal([10.0, 1.0])

    N_horizon = 30
end


# ╔═╡ a7c85dc4-c70a-4383-b4b1-d1fc017ca0de
begin
    """
        finite_horizon_lqr(A,B,Q,R,Qf,N)

    Compute finite-horizon LQR gains K[1:N] and value matrices P[1:N+1].
    """
    function finite_horizon_lqr(A,B,Q,R,Qf,N)
        n = size(A, 1)
        m = size(B, 2)
        P = Vector{Matrix{Float64}}(undef, N + 1)
        K = Vector{Matrix{Float64}}(undef, N)

        P[N + 1] = Qf
        for k in N:-1:1
            Pnext = P[k + 1]
            S = R + B' * Pnext * B
            K[k] = inv(S) * (B' * Pnext * A)
            P[k] = Q + A' * Pnext * A - A' * Pnext * B * K[k]
        end
        return K, P
    end

    K_lqr, P_lqr = finite_horizon_lqr(A,B,Q,R,Qf,N_horizon)
    @info "First-step feedback gain K₀" K_lqr[1]
end


# ╔═╡ 7cbc7743-9564-48d6-9250-7550e78221b8
begin
    """
        simulate_lqr(A,B,K,x0)

    Simulate closed-loop system x_{k+1} = (A - B K_k) x_k
    with state-feedback u_k = -K_k x_k.
    """
    function simulate_lqr(A,B,K,x0)
        N = length(K)
        n = length(x0)
        m = size(B, 2)

        xs = zeros(n, N + 1)
        us = zeros(m, N)

        xs[:, 1] .= x0
        for k in 1:N
            us[:, k] .= -K[k] * xs[:, k]
            xs[:, k + 1] .= A * xs[:, k] + B * us[:, k]
        end
        return xs, us
    end

    x0_lqr = [2.0, 0.0]
    xs_lqr, us_lqr = simulate_lqr(A,B,K_lqr,x0_lqr)

    ks = 0:N_horizon

    fig_lqr = Figure(resolution = (900, 400))

    ax1_lqr = Axis(fig_lqr[1, 1],
                   title = "Finite-horizon LQR state trajectory",
                   xlabel = "k", ylabel = "x")
    lines!(ax1_lqr, ks, xs_lqr[1, :], label = "position")
    lines!(ax1_lqr, ks, xs_lqr[2, :], label = "velocity")
    axislegend(ax1_lqr, position = :rt)

    ax2_lqr = Axis(fig_lqr[1, 2],
                   title = "Finite-horizon LQR control sequence",
                   xlabel = "k", ylabel = "u")
    lines!(ax2_lqr, ks[1:end-1], us_lqr[1, :], label = "u_k")
    axislegend(ax2_lqr, position = :rt)

    fig_lqr
end


# ╔═╡ 2f3b4c65-b445-49f1-ba54-5c4951c175c5
md"
## Dynamic programming viewpoint

Assume the value function has quadratic form

```math
V_k(x) = \frac12 x^\top P_k x.
```

The Bellman equation is

```math
V_k(x)
= \min_u \left(
    \frac12 x^\top Q x
  + \frac12 u^\top R u
  + V_{k+1}(A x + B u)
\right).
```

Substituting the quadratic ansatz and expanding leads to the Riccati recursion above.
"

# ╔═╡ f9929ed7-28f6-4a5e-b102-c448ddcf027d
question_box(md"
**Q3. How is the Riccati recursion related to the Bellman equation for LQR?**
")

# ╔═╡ 84b1c080-27eb-4ee2-88b8-d23db8e40cb1
Foldable(md"Q3 – Sketch of the derivation",
md"
Plug the quadratic ansatz

```math
V_{k+1}(x) = \frac12 x^\top P_{k+1} x
```

into the Bellman equation. The minimization over \(u\) is equivalent to
minimizing a quadratic form

Computing the minimizer gives

```math
u_k^\* = -K_k x,
```

and the minimal value has Hessian \(P_k\), which satisfies the Riccati update.
This shows that the Riccati recursion is the algebraic form of dynamic programming
for the LQR problem.
")

# ╔═╡ 1c93c0bd-12a0-4395-9063-4a52c9effb99
md"
## QP viewpoint

Stack all states and inputs:

```math
\mathbf{x} =
\begin{bmatrix}
x_0 \\ x_1 \\ \vdots \\ x_N
\end{bmatrix},
\qquad
\mathbf{u} =
\begin{bmatrix}
u_0 \\ u_1 \\ \vdots \\ u_{N-1}
\end{bmatrix}.
```

We can write dynamics constraints as

```math
E \mathbf{x} + F \mathbf{u} = g,
```

and the cost as

```math
J = \frac12 \mathbf{z}^\top H \mathbf{z},
\qquad
\mathbf{z} =
\begin{bmatrix}
\mathbf{x} \\
\mathbf{u}
\end{bmatrix}.
```

This yields a convex quadratic program whose KKT conditions are equivalent to
the Riccati recursion [^bertsekas].
"


# ╔═╡ b8cc191d-01a1-43c7-a4ed-8cd6918c11d4
begin
    """
        build_lqr_qp2(A,B,Q,R,Qf,N,x0)

    Build a JuMP model for the finite-horizon LQR as a QP (version 2).
    """
    function build_lqr_qp2(A,B,Q,R,Qf,N,x0)
        n = size(A, 1)
        m = size(B, 2)

        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, x_qp2[1:n, 0:N])
        @variable(model, u_qp2[1:m, 0:N-1])

        # Initial condition
        @constraint(model, x_qp2[:, 0] .== x0)

        # Dynamics
        for k in 0:N-1
            @constraint(model, x_qp2[:, k+1] .== A * x_qp2[:, k] + B * u_qp2[:, k])
        end

        # Cost expression
        @expression(model, cost_qp2,
            0.5 * (x_qp2[:, N]' * Qf * x_qp2[:, N]) +
            0.5 * sum(x_qp2[:, k]' * Q * x_qp2[:, k] for k in 0:N-1) +
            0.5 * sum(u_qp2[:, k]' * R * u_qp2[:, k] for k in 0:N-1)
        )
        @objective(model, Min, cost_qp2)

        return model, x_qp2, u_qp2
    end

    # Build and solve the QP (version 2)
    model_qp2, x_qp2, u_qp2 = build_lqr_qp2(A,B,Q,R,Qf,N_horizon,x0_lqr)
    optimize!(model_qp2)
    @info "QP termination status (v2)" termination_status(model_qp2)
end

# ╔═╡ 1031673a-9e7c-4d93-b102-3db1e22625b8
begin
    # Extract trajectories from QP solution (version 2)
    xs_qp2 = [value.(x_qp2[:, k]) for k in 0:N_horizon]
    us_qp2 = [value.(u_qp2[:, k]) for k in 0:N_horizon-1]

    xs_qp2_mat = hcat(xs_qp2...)
    us_qp2_mat = hcat(us_qp2...)

    fig_cmp2 = Figure(resolution = (900, 400))

    ax1_cmp2 = Axis(fig_cmp2[1, 1],
                    title = "State: Riccati vs QP (v2)",
                    xlabel = "k", ylabel = "x₁")
    lines!(ax1_cmp2, 0:N_horizon, xs_lqr[1, :], label = "x₁ (Riccati)")
    lines!(ax1_cmp2, 0:N_horizon, xs_qp2_mat[1, :],
           linestyle = :dash, label = "x₁ (QP v2)")
    axislegend(ax1_cmp2, position = :rt)

    ax2_cmp2 = Axis(fig_cmp2[1, 2],
                    title = "Control: Riccati vs QP (v2)",
                    xlabel = "k", ylabel = "u")
    lines!(ax2_cmp2, 0:N_horizon-1, us_lqr[1, :], label = "u (Riccati)")
    lines!(ax2_cmp2, 0:N_horizon-1, us_qp2_mat[1, :],
           linestyle = :dash, label = "u (QP v2)")
    axislegend(ax2_cmp2, position = :rt)

    fig_cmp2
end



# ╔═╡ Cell order:
# ╠═f65b22ca-183c-4cd2-a9af-0b47e90df929
# ╠═30fe161b-83af-4199-bf90-6a188d632738
# ╠═ed800160-c3ed-11f0-0aff-9b9e6e4a3d5a
# ╠═8b5bf689-a835-49df-82d3-f0de8ed0d444
# ╠═1c00dbc5-36d6-4fdd-bb8b-2c171a934653
# ╠═af09aa0e-88bd-4f6d-93b3-c35144133fb5
# ╠═8b31afac-acad-46df-a213-6921567cce04
# ╠═55c540b6-f463-4628-b5f8-5b835d15c3c7
# ╠═8422c150-e56c-4e60-a8ed-9aef1af39339
# ╠═647a107a-1fb1-4a30-8ea5-57fbcdec96d3
# ╠═c47669aa-71c0-49b0-9185-7eacf79ebb25
# ╠═8fda68ae-6a02-4890-945a-5054c51c53e4
# ╠═a2566189-e807-4a08-ae0a-56fbe92a19c9
# ╠═0659e29f-c6e0-4881-99b9-25a21214af37
# ╠═202312d6-89c6-430b-ad21-63397c7f57b7
# ╠═5a80124b-8671-4262-9178-cb1696d244ca
# ╠═780a258c-8e2d-4822-ae0a-dc2454574009
# ╠═c6ffb973-b201-47f5-88c3-d52ad5dda5cd
# ╠═be0ab297-eeee-471b-96a0-3fa7301be244
# ╠═95cfd865-6c84-46a9-8ef7-2c2cc241740d
# ╠═a7c85dc4-c70a-4383-b4b1-d1fc017ca0de
# ╠═7cbc7743-9564-48d6-9250-7550e78221b8
# ╠═2f3b4c65-b445-49f1-ba54-5c4951c175c5
# ╠═f9929ed7-28f6-4a5e-b102-c448ddcf027d
# ╠═84b1c080-27eb-4ee2-88b8-d23db8e40cb1
# ╠═1c93c0bd-12a0-4395-9063-4a52c9effb99
# ╠═b8cc191d-01a1-43c7-a4ed-8cd6918c11d4
# ╠═1031673a-9e7c-4d93-b102-3db1e22625b8
