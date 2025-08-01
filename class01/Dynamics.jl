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

# ╔═╡ 13b12c00-6d6e-11f0-3780-a16e73360478
begin
	import Pkg
	Pkg.activate(".")
	# Pkg.status()
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
end

# ╔═╡ b6ba1231-2942-4f06-8252-22f02553bb57
using CairoMakie

# ╔═╡ ec473e69-d5ec-4d6a-b868-b89dadb85705
ChooseDisplayMode()

# ╔═╡ 52005382-177b-4a11-a914-49a5ffc412a3
md"# 101 (Continuous-Time) Dynamics
#### A Crash Course

General form for a smooth system:

```math
\dot{x} = f(x,u) \quad \text{First-Order Ordinary Differential Equation (ODE)}
```
where
```math
\begin{cases}
f: \mathbb{R}^{n} \times \mathbb{R}^{m} \rightarrow \mathbb{R}^{n} & \text{Dynamics} \\
x \in \mathbb{R}^{n} & \text{State} \\
u \in \mathbb{R}^{m} & \text{Control} \\
\dot{x} \in \mathbb{R}^{n} & \text{Time derivative of } x \\
\end{cases}
```

"

# ╔═╡ 2be161cd-2d4c-4778-adca-d45f8ab05f98
Foldable(md"What would $F=ma$ be?", md"""

A $2^{\text{nd}}$--Order ODE! But we can always write them as $1^{\text{st}}$--Order.
For a mechanical system:

```math
x=\begin{bmatrix}
q \\
\dot{q}=v
\end{bmatrix} \implies 
\dot{x}=\begin{bmatrix}
v \\
\dot{v}=a
\end{bmatrix} 
```
where
```math
\begin{cases}
q & \text{Configuration/Pose} \\
v & \text{Velocity/Angular-Velocity}
\end{cases}
```

**$q$ is not always a vector -- but a `Lie group / Differentiable Manifold`. Examples?**

Even if $q$ is not a vector, $v$ is!
		 
""")

# ╔═╡ b452ee52-ee33-44ad-a980-6a6e90954ee1
md"State $x$ is everything you need to define to determine the how your system will progress through time--The initial conditions / time-varying constraints of your problem.
"

# ╔═╡ 9f62fae9-283c-44c3-8d69-29bfa90faf29
md"### Example: Pendulum"

# ╔═╡ baa3993c-96b0-474e-b5b4-f9eaea642a49
function pendulum(θ_deg = 60; L = 4, fsize = (520, 450))
    θ       = deg2rad(θ_deg)
    pivot   = Point2f(0, 0)
    mass    = Point2f(-L*sin(θ), -L*cos(θ))        # rod tip
    rodϕ    = -π/2 -θ               # rod’s actual angle (≈ -120° here)

    fig = Figure(size = fsize)
    ax  = Axis(fig[1, 1];
               aspect         = 1,
               xticksvisible  = false,
               yticksvisible  = false,
               xgridvisible   = false,
               ygridvisible   = false)
    hidespines!(ax)

	_y = -L*cos(θ)
	if _y > 0 
		ylims!(ax, (-1, _y + 1))
	end

    ## ceiling ------------------------------------------------------------------
    lines!(ax, [-5,  5], [0, 0]; linewidth = 3)
    foreach(x -> lines!(ax, [x, x], [0, 0.4]; linewidth = 2), -4.5:1:4.5)

    ## vertical reference -------------------------------------------------------
    lines!(ax, [0, 0], [0, -L - 1]; linestyle = :dash)

    ## rod + “ℓ” ----------------------------------------------------------------
    lines!(ax, [pivot[1], mass[1]], [pivot[2], mass[2]]; linewidth = 3)
    mid = 0.6 .* (pivot .+ mass) .+ Point2f(0.25, 0.5)
    text!(ax, mid, text = "ℓ", fontsize = 18)

    ## angle arc ---------------------------------------------------------------
    r  = 0.2L
    ts = range(-π/2, rodϕ; length = 60)             # sweep **toward the rod**
    lines!(ax, r .* cos.(ts), r .* sin.(ts); linewidth = 2)
    text!(ax, Point2f(r*0.05, -0.9r), text = "θ", fontsize = 18)

    ## mass ---------------------------------------------------------------------
    scatter!(ax, [mass]; markersize = 55, color = :white, strokewidth = 3)
    text!(ax, mass, text = "m", align = (:center, :center))

    fig
end

# ╔═╡ 9ec1f918-ff16-4a94-b75f-4b07e2931d4c
@bind θ PlutoUI.Slider(0:1:360, default = 60, show_value = x-> "θ = $(x)")

# ╔═╡ 2f42d32e-8e53-458a-816e-292861a8b8ef
pendulum(θ) 

# ╔═╡ ab369bb9-ecce-4c7b-b082-d6ae49beafe8
Foldable(md"How do we write the dynamics?", md"""

The $2^{\text{nd}}$--Order ODE:		 
```math
m \cdot l^{2} \cdot \ddot{\theta} + m \cdot g \cdot l \cdot sin(\theta) = \tau
```
where
```math
\begin{cases}
m & \text{Mass} \\
l & \text{Length of the pole} \\
\theta & \text{Pole angular position} \\
g & \text{Gravity} \\
\tau & \text{Torque exerted at axis}
\end{cases}
``` 

""")

# ╔═╡ bd1b6301-0b4d-4f94-81bb-e0267792aca0
Foldable(md"How to write it as a $1^{\text{st}}$--Order ODE:?", md"""

```math
x=\begin{bmatrix}
\theta \\
\dot{\theta}
\end{bmatrix} \implies 
\dot{x}=\begin{bmatrix}
\dot{\theta} \\
\ddot{\theta}
\end{bmatrix} =
\begin{bmatrix}
\dot{\theta} \\
\frac{-g sin(\theta)}{l} + \frac{\tau}{ml^{2}}
\end{bmatrix}
```
**Angles are not in** $\mathbb{R}$! In fact:
```math
\begin{cases}
e^{i\theta} \in S^{1} & \text{Configuration in the Circle Group} \\
\dot{\theta} \in \mathbb{R} \\
x \in S^{1} \times \mathbb{R} & \text{Cylinder}
\end{cases}
```

""")

# ╔═╡ 4d598933-05a9-44fa-b5a7-f7e1c7afb094
md"## Control--Afine Systems

Non--linear Systems of the form:
```math
\dot{x} = \underbrace{f_{o}(x)}_{\text{drift}} +  \underbrace{B(x)}_{\text{input Jacobian}}u
```

 $\implies$ Non--linear in the state but affine in the input/control.

Control--Afine Systems are common in many mechanical systems.

"

# ╔═╡ 5f408845-7870-425b-af53-b9e2a8d0c2ea
Foldable(md"Pendulum?", md"""

```math
f_{o}(x)=\begin{bmatrix}
\dot{\theta} \\
\frac{-g sin(\theta)}{l}
\end{bmatrix},\quad 
B(x)=\begin{bmatrix}
0 \\
\frac{1}{ml^{2}}
\end{bmatrix}
```
""")

# ╔═╡ 962b427e-3712-4b7f-b971-5c29be434dca
Foldable(md"What happens if $B(x)$ is full rank?", md"""

Habemus a fully--actuated system! We can easily solve for $u$:

```math
u = B(x)^{-1}(\dot{x} - f_{o}(x))
```

> A system where the number of actuators (or control inputs) is equal to the number of degrees of freedom (DOF) of the system.

See **Feedback linearization** approaches.
""")

# ╔═╡ f10927fe-d392-4374-bad1-ab5ac85b8116
md"## Manipulator Dynamics

```math
\begin{cases}
M(q) \dot{v} +  C(q,v) + B(q)u + F \\
\dot{q} = G(q)v & \text{(Velocity Kinematics)}
\end{cases} \qquad \qquad \qquad \qquad \qquad \qquad \qquad 
```
```math
\qquad \implies
\qquad \dot{x} = f(x,u) =\begin{bmatrix}
G(q)v \\
M(q)^{-1}(B(q)u + F - C(q,v))
\end{bmatrix}
```
where
```math
\begin{cases}
M(q) & \text{Mass Matrix / Generalized Inertia Tensor} \\
C(q,v) & \text{Dynamics Bias (Corriolis, Gravity)} \\
B(q) & \text{Input Jacobian} \\
F & \text{External Forces}
\end{cases}
```
"

# ╔═╡ b8b206ef-cdc5-4cc9-9b55-70d711ba2a9e
Foldable(md"Pendulum?", md"""

```math
M(q) = ml^2, \; C(q,v) = mgl\sin(\theta), \; B=I, \; G=I 
```
""")

# ╔═╡ a09de9e4-7ecc-4d23-9135-384077f0c03f
Foldable(md"All mechanical systems can be written this way. Why?", md"""

Manipulator Dynamics Equations are a way of rewriting the Euler--Lagrange equations.

> In the calculus of variations and classical mechanics, the Euler–Lagrange equations are a system of second-order ordinary differential equations whose solutions are stationary points of the given action functional. The equations were discovered in the 1750s by Swiss mathematician Leonhard Euler and Italian mathematician Joseph-Louis Lagrange.

```math
L = \underbrace{\frac{1}{2} v^{\top}M(q)v}_{\text{Kinematic Energy}} - \underbrace{U(q)}_{\text{Potential Energy}}
```

What can you say about $M(q)$? When do we have a problem inverting it?

""")

# ╔═╡ 5f35a169-887f-477f-b010-167627f7ce4c
md"## Linear Systems

```math
\dot{x} = A_{t}x + B_{t}u
```

When Time--Invariant vs Time--Variant?

Non--Linear Systems are often approximated by Linear Systems (locally).
"

# ╔═╡ e860d92b-cc8f-479b-a0fc-e5f7a11ae1fd
Foldable(md" $\dot{x} = f(x,u) \; \implies \; A=? \; B=?$", md"""

```math
A= \frac{\partial f}{\partial x}, \quad A= \frac{\partial f}{\partial u}
```

""")

# ╔═╡ bb4bfa72-bf69-41f5-b017-7cbf31653bae
Foldable(md"Why approximate? What happens to the optimal control problem?", md"""

The problem becomes convex!!

""")

# ╔═╡ 2936c97e-a407-4e56-952f-0a2dfb7acf83
md"""## Equilibria

A point at which the system is and will remain at "rest":

```math
\dot{x} = f(x,u) = 0
```

The root of the dynamic equations!

"""

# ╔═╡ 1a154d04-2b33-43b6-9cb6-accd935de7b7
Foldable(md"Pendulum?", md"""

```math
\dot{x} =
\begin{bmatrix}
\dot{\theta} \\
\frac{-g sin(\theta)}{l}
\end{bmatrix}=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\implies
\begin{cases}
\dot{\theta} = 0 & \text{No Velocity} \\
\theta = 0, \; \pi, \dots
\end{cases}
```
$([pendulum(0; fsize=(250,250), L=4), pendulum(180; fsize=(250,250), L=4)])
""")

# ╔═╡ 593e2764-7e77-4756-ae62-cfc3eb039444
question_box(md"### Can I use control to move the equilibria?")

# ╔═╡ 17939d59-1ba1-483c-864c-fed049b54151
Columns(md"""

How about if I want $\theta = \pi / 2$ ?

```math
\begin{cases}
\theta = \pi / 2 \\
\dot{x} =
\begin{bmatrix}
\dot{\theta} \\
\frac{-g sin(\theta)}{l} + \frac{\tau}{ml^{2}}
\end{bmatrix}=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\end{cases}
```
```math
\implies \frac{\tau}{ml^{2}} = \frac{g sin(\pi / 2)}{l}
```
```math
\implies u = m\,g\,l
```
""",
pendulum(90; fsize=(250,250), L=4)
)
		

# ╔═╡ aa63e35d-13dd-4910-b2fd-be017cda4b55
md"
In general, we get a root finding problem in u:

```math
f(x^{*},u) = 0
```

> You can see control as changing a vector filed into a chosen dynamics[^cmu]
"

# ╔═╡ b180beb7-9606-4332-8e94-cd4546b4bc59
md"""
## Stability of Equilibria

When will the system stay "near" an equilibrium point under pertubations?
"""

# ╔═╡ 0e29ab58-e56c-4f54-aa2a-3152034ddc0c
md"### 1--D System"

# ╔═╡ d0d251ec-4ea9-417a-90c2-3f19f4b75aa8
md"""
 Outer points: $(@bind var1 CheckBox()) | Inner: $(@bind var2 CheckBox())
"""

# ╔═╡ 4f69216c-fc31-45d5-9699-c774f9f77a24
begin
	import Plots: plot, hline!, vline!, plot!
	f(x) = x^3 - 3*x
	plt = plot(range(-2.2,2.2, 1000),f, label="ẋ = x³ - 3*x", xlabel="x",
			   ylabel="ẋ")
	hline!(plt, [0], label="", color=:black, style=:dash)
	vline!(plt, [0], label="", color=:black, style=:dash)
	if var2
		plot!(plt, [0.5,0.1], [0.2, 0.2],arrow=true,color=:green,linewidth=2,label="")
		plot!(plt, [-0.5,-0.1], [0.2, 0.2],arrow=true,color=:green,linewidth=2,label="")
	end
	if var1
		plot!(plt, [1.9,2.4], [0.2, 0.2],arrow=true,color=:red,linewidth=2,label="")
		plot!(plt, [1.6,1.1], [0.2, 0.2],arrow=true,color=:red,linewidth=2,label="")
		plot!(plt, [-1.9,-2.4], [0.2, 0.2],arrow=true,color=:red,linewidth=2,label="")
		plot!(plt, [-1.6,-1.1], [0.2, 0.2],arrow=true,color=:red,linewidth=2,label="")
	end
	plt
end

# ╔═╡ f659d05c-e345-46c8-9c7b-c1adf95c9023
Foldable(md"Can we say anything about the slope of $\dot{x}$?", md"""

```math
\begin{cases}
\frac{\partial f}{\partial x} < 0 & \text{Stable} \\
\frac{\partial f}{\partial x} > 0 & \text{Unstable} 
\end{cases}
```

""")

# ╔═╡ 7dc0c8c9-ba46-43ab-a7e3-c2e160be141c
md"### Basin of Attraction

> The set of all points in the phase space that are attracted to a specific equilibrium point (or attractor). "

# ╔═╡ 25bfc51e-11cf-48f6-9b92-9ac682db05a8
Foldable(md"What is the $\textit{Basin of Attraction}$ of $(0,0)$ ?", md"""

The space between the unstable equilibrium points:
		 
```math
\{ x | - \sqrt{3} < x < \sqrt{3} \}
```

""")

# ╔═╡ 876bdea3-9a0e-4e40-9ae4-ef77b08c2428
Foldable(md"What if the slope was 0?", md"""


		 TODO: Lyapunov stability
```math

```

""")

# ╔═╡ 161a2a6e-567f-4994-8d77-9a0f0962cdd9
md"""
As we increase the dimensions, it gets increasingly more complicated to reason about how a system will evolve!

> For continuous, autonomous dynamical systems, the Poincaré–Bendixson theorem states that chaos cannot occur in phase spaces of dimension less than 3.
"""

# ╔═╡ 97994ed8-5606-46ef-bd30-c5343c1d99cf
begin
	MarkdownLiteral.@markdown(
"""

[^cmu]: [CMU Course on Optimal Control]("https://optimalcontrol.ri.cmu.edu/")

"""
)
end

# ╔═╡ Cell order:
# ╟─13b12c00-6d6e-11f0-3780-a16e73360478
# ╟─ec473e69-d5ec-4d6a-b868-b89dadb85705
# ╟─52005382-177b-4a11-a914-49a5ffc412a3
# ╟─2be161cd-2d4c-4778-adca-d45f8ab05f98
# ╟─b452ee52-ee33-44ad-a980-6a6e90954ee1
# ╟─9f62fae9-283c-44c3-8d69-29bfa90faf29
# ╠═b6ba1231-2942-4f06-8252-22f02553bb57
# ╟─baa3993c-96b0-474e-b5b4-f9eaea642a49
# ╟─9ec1f918-ff16-4a94-b75f-4b07e2931d4c
# ╟─2f42d32e-8e53-458a-816e-292861a8b8ef
# ╟─ab369bb9-ecce-4c7b-b082-d6ae49beafe8
# ╟─bd1b6301-0b4d-4f94-81bb-e0267792aca0
# ╟─4d598933-05a9-44fa-b5a7-f7e1c7afb094
# ╟─5f408845-7870-425b-af53-b9e2a8d0c2ea
# ╟─962b427e-3712-4b7f-b971-5c29be434dca
# ╟─f10927fe-d392-4374-bad1-ab5ac85b8116
# ╟─b8b206ef-cdc5-4cc9-9b55-70d711ba2a9e
# ╟─a09de9e4-7ecc-4d23-9135-384077f0c03f
# ╟─5f35a169-887f-477f-b010-167627f7ce4c
# ╟─e860d92b-cc8f-479b-a0fc-e5f7a11ae1fd
# ╟─bb4bfa72-bf69-41f5-b017-7cbf31653bae
# ╟─2936c97e-a407-4e56-952f-0a2dfb7acf83
# ╟─1a154d04-2b33-43b6-9cb6-accd935de7b7
# ╟─593e2764-7e77-4756-ae62-cfc3eb039444
# ╟─17939d59-1ba1-483c-864c-fed049b54151
# ╟─aa63e35d-13dd-4910-b2fd-be017cda4b55
# ╟─b180beb7-9606-4332-8e94-cd4546b4bc59
# ╟─0e29ab58-e56c-4f54-aa2a-3152034ddc0c
# ╟─d0d251ec-4ea9-417a-90c2-3f19f4b75aa8
# ╟─4f69216c-fc31-45d5-9699-c774f9f77a24
# ╟─f659d05c-e345-46c8-9c7b-c1adf95c9023
# ╟─7dc0c8c9-ba46-43ab-a7e3-c2e160be141c
# ╟─25bfc51e-11cf-48f6-9b92-9ac682db05a8
# ╠═876bdea3-9a0e-4e40-9ae4-ef77b08c2428
# ╟─161a2a6e-567f-4994-8d77-9a0f0962cdd9
# ╟─97994ed8-5606-46ef-bd30-c5343c1d99cf
