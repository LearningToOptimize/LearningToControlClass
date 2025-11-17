### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 2fe513ba-9310-11f0-2266-9730fc13e5da
begin
	class_dir = @__DIR__
	import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using PlutoUI
	using PlutoTeachingTools
	using MarkdownLiteral
end

# ╔═╡ e9f70acf-cadd-43a9-9068-0931e24f595c
begin
	using JuMP
	using Ipopt
	using CairoMakie
end

# ╔═╡ 4de1acea-66ba-4e82-9426-8f3e152c3998
begin
	using TORA
	using MeshCat
	using RigidBodyDynamics
end

# ╔═╡ e6aa5227-91bd-4cec-9448-24384708a305
ChooseDisplayMode()

# ╔═╡ 73e305d8-6619-4226-81cd-ee7a2fb59c3d
md"
| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Qiu, Guancheng \"Ivan\" |
|  Date   | : | 19 of September, 2025 |
"

# ╔═╡ d0c88449-5cb8-4cb2-9223-aae0482d240f
md"

```math
\newcommand{\bu}{\mathbf{u}}
\newcommand{\bx}{\mathbf{x}}
```

"

# ╔═╡ 27b4a8c3-bf81-4df7-a640-5037e947595f
md"
#### Reference Material
"

# ╔═╡ 7899bee5-fdaf-4929-8073-c36d16c2e24f
begin
	MarkdownLiteral.@markdown(
"""
[^cmu11]: Z. Manchester, Optimal Control (CMU 16-745) 2025 Lecture 11: Nonlinear Trajectory Optimization, Carnegie Mellon University, YouTube, 2025. [Online]. Available: [https://youtu.be/ERGKQ2ieYW8](https://youtu.be/ERGKQ2ieYW8)

[^cmu13]: Z. Manchester, Optimal Control (CMU 16-745) 2025 Lecture 13: Direct Trajectory Optimization, Carnegie Mellon University, YouTube, 2025. [Online]. Available: [https://youtu.be/8VZ0MZ_JpgE](https://youtu.be/8VZ0MZ_JpgE)

[^kelly2017]: M. P. Kelly, “An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation,” SIAM Review, vol. 59, no. 4, pp. 849–904, Dec. 2017.
		
[^li2024]: F. Li, A. Abuduweili, Y. Sun, R. Chen, W. Zhao and C. Liu, “Continual Learning and Lifting of Koopman Dynamics for Linear Control of Legged Robots,” arXiv preprint arXiv:2411.14321, Nov. 2024.
	
[^mit1]: R. Tedrake, Ch. 1: Fully-actuated vs Underactuated Systems, Underactuated Robotics, MIT, 2024. [Online]. Available: [https://underactuated.mit.edu/intro.html](https://underactuated.mit.edu/intro.html)
		
[^mit10]: R. Tedrake, Ch. 10: Trajectory Optimization, Underactuated Robotics, MIT, 2024. [Online]. Available: [https://underactuated.mit.edu/trajopt.html](https://underactuated.mit.edu/trajopt.html)

[^tora]: H. Ferrolho et al., Tutorial: TORA.jl — Trajectory Optimization for Robot Arms, JuliaRobotics, 2025. [Online]. Available: [https://juliarobotics.org/TORA.jl/stable/tutorial/](https://juliarobotics.org/TORA.jl/stable/tutorial/)
"""
)
end

# ╔═╡ bfc7cced-3ce7-4f2b-8ee9-424d6d5ba682
md"
For systems with linear dynamics, if quadratic functions are good enough to represent the cost (which often is the case), then the trajectory optimization problem can be modeled as LQR (refer to Lecture 3).
However, the real world is hardly linear, and a linear approximation of the dynamics could prove insufficient given the specific setting of the problem.

# Nonlinear Trajectory Optimization

```math
\begin{align*}
\min_{u(\cdot)} \quad & \ell_f (x(t_f)) + \int_{t_0}^{t_f} \ell(x(t), u(t)) dt \\
\text{s.t.} \quad
    & \dot{x}(t) = f(x(t),u(t)), \quad \forall t \in [t_0, t_f] \\
	& \bx(t_0) = x_0 \\
	& ...
\end{align*}
```

This trajectory optimization problem is often nonlinear in practice.
Nonlinear dynamics causes the problem to be nonconvex. Nonlinearity could also arise from additional constraints, not just the dynamics.
"

# ╔═╡ 5f190a4e-d4b6-4279-8757-b0ec89df987f
begin
md"
### Examples
##### Actuated pendulum
```math
	\dot{x} = \begin{pmatrix} \dot{\theta} \\ \ddot{\theta} \end{pmatrix} = \begin{pmatrix} \dot{\theta} \\ \frac{-g \sin(\theta)}{l} + \frac{u}{m l^{2}} \end{pmatrix}
```

What about a pendulum in space (no gravity)?
	
"
end

# ╔═╡ 0de87fc1-7764-4194-8279-7c86fb3c64b8
md"
##### The Acrobot
The Acrobot is a robotic arm with only an actuator in the elbow.
"

# ╔═╡ f49e647c-529d-4615-8bd2-7e18a7521eea
html"""

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/FeCwtvrD76I?rel=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


"""

# ╔═╡ dd00db44-8b37-44d6-9ace-0411c62e1dad
md"
The Acrobot is an example of **underactuated** systems. Underactuated systems, simply put, are systems that have fewer actuators than degrees of freedom. The Acrobot has only 1 actuator but 2 degrees of freedom.

Underactuation often introduces coupling in the dynamics of different parts of a system, giving rise to nonlinearity.

Underactuated robotics is a promising growing field [^mit1], with progress displayed by, e.g., the success of Boston Dynamics.
"

# ╔═╡ 9876a377-0da0-44a6-8922-378639724199
md"""
##### Unicycle robot
![unicycle](https://raw.githubusercontent.com/wheelbot/Mini-Wheelbot/refs/heads/main/imgs/circle.gif)

Recall the system dynamics

```math
x = \begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}
:=
\begin{bmatrix}
p_x\cos\theta + (p_y-1)\sin\theta \\[4pt]
-\,p_x\sin\theta + (p_y-1)\cos\theta \\[4pt]
\theta
\end{bmatrix}
```

```math
\dot x =
\begin{bmatrix}
v + \omega x_2 \\[2pt]
-\omega x_1 \\[2pt]
\omega
\end{bmatrix}
```
where $u = [v,\,\omega]^\top$.
"""

# ╔═╡ 626d09d9-4f46-427f-9b91-a95ba58a90da
question_box(md"Does the unicycle have linear dynamics?")

# ╔═╡ e3efd893-2d8f-42ae-93f4-85a5997ceee8
Foldable(md"Answer...", md"
No, $\dot{x}$ is not linear in $x$ and $u$.
")

# ╔═╡ 8af4baab-a665-4111-b93f-d85ef8293ae8
md"
The dynamics of the unicycle is called **nonholonomic**, meaning that the ways in which it reaches a certain configuration are restricted. For example, it can't side-slip to a position on its right side.

Nonholonomic dynamics often introduces coupling in configuration and velocity, and therefore nonlinearity.
"

# ╔═╡ 4b470369-ef0b-49bc-b013-05235eb28230
md"
##### Manipulator dynamics

Consider a more general example. Recall the manipulator dynamics from Lecture 1:
```math
\begin{cases}
M(q) \dot{v} +  C(q,v) + B(q)u + F = 0 \\
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

If $M(q)$, $B(q)$, $G(q)$ are not constant and $C(q,v)$ is not linear in $q$ and $v$, we have nonlinear dynamics.
"

# ╔═╡ 1932a4b1-36fd-4f26-9a00-f7ea609791ae
md"
##### Friction cone
Nonlinearity may show up in other constraints too, not just in dynamics.
For legged robots, a constraint of the form
```math
||f||_{2} \leq \mu f_{N}
```
may be enforced to prevent slipping, where $f$ is the applied force, $\mu$ is the coefficient of statis friction and $f_{N}$ is the normal force.
"

# ╔═╡ 055ac28a-2fbd-4777-b683-688ae6b10a89
Foldable(md"Model choice: when does a linear controller suffice in robotics?", md"
> If you're not pushing the performance limit, then you can probably use a linear model. [^cmu11]

In a recent paper [^li2024], legged robots are controlled with linear controllers using data-driven Koopman linearization to walk.
![unitree](https://arxiv.org/html/2411.14321v3/x3.png)
")

# ╔═╡ a728d0f1-6c81-4342-9628-6957acbfda73
md"
We will now explore two of the many techniques for solving the nonlinar trajectory optimization problem:
- **Direct transcription**: formulate a discretized version of the original problem into a finite-dimensional optimization problem.
- **Differential dynamic programming**: an algorithm similar to approximate dynamic programming designed specifically for solving nonlinear trajectory optimization.
"

# ╔═╡ 359ae83f-af31-4363-bb35-19ec33586eae
md"
## Direct Transcription

**Discretize the trajectory optimization into a finite-dimensional optimization problem** (that can be directly solved by an optimization solver).

```math
\begin{align*}
\min_{x[\cdot], u[\cdot]} \quad & \ell_f (x[N]) + \sum_{n=0}^{N-1} h_{n} \ell(x[n], u[n]) \\
\text{s.t.} \quad
    & \text{(dynamics constraints)} \\
	& x[0] = x_0 \\
	& ...
\end{align*}
```

(When applied to problems with linear dynamics, direct transcription yields a convex program.)
"

# ╔═╡ 02589859-8f35-4b08-86f5-59698c6a6224
md"
#### Direct Collocation
A special case of direct transcription which **approximates the $x$ and $u$ trajectories with splines (piecewise polynomials)**.

Discretize by setting **break points** in time which the splines are fit on.

A common choice: using a cubic spline for $x$ and a linear spline for $u$.
This is known as the Hermite-Simpson collocation. \"Hermite\" comes from \"cubic Hermite spline\". \"Simpson\" will be explained later.


In this method, we would like to use a cubic spline to model the $x$ trajectory, so we will not enforce something like explicit Euler integration $x[k+1] = x[k] + h_{k} f(x[k], u[k])$ which would be a linear approximation.

Instead, we enforce a dynamics constraint at a point within every time interval,a **collocation point**, often taken to be
```math
t_{k + \frac{1}{2}} = \frac{1}{2} (t_{k} + t_{k+1})
```
This makes sure that in each time interval $(t_{k}, t_{k+1})$, the approximate trajectory of $x$ is roughly consistent with the real dynamics.

![collocation](https://underactuated.mit.edu/data/collocation.svg) [^mit10]

The constraints will just be
```math
\dot{x}(t_{k + \frac{1}{2}}) = f(x(t_{k + \frac{1}{2}}), u(t_{k + \frac{1}{2}}))
```

The optimization problem is
```math
\begin{align*}
\min_{x[\cdot], u[\cdot]} \quad & \ell_f (x[N]) + \sum_{k=0}^{N-1} h_{k} \ell(x[k], u[k]) \\
\text{s.t.} \quad
    & \dot{x}(t_{k + \frac{1}{2}}) = f(x(t_{k + \frac{1}{2}}), u(t_{k + \frac{1}{2}})), \quad \forall k \in \{1, 2, ..., N - 1\} \\
	& x[0] = x_0 \\
	& ...
\end{align*}
```

How to express these variables at the collocation points? If we know the splines, then we can express them. We will explore how to do this in the following part.
"

# ╔═╡ 165297a6-854f-475c-a16a-637de6dc9b69
question_box(md"Why do you think one would choose cubic spline for $x$ and linear spline for $u$?")

# ╔═╡ a193f3b5-5814-4528-9b07-92cdd493bd05
question_box(md"If we have the explicit representation of the cubic spline $x(t) = C_{0} + C_{1}t + C_{2} t^{2} + C_{3} t^{3}$, then $x(t_{k + \frac{1}{2}})$ can be expressed in terms of the polynomial coefficients $C$. Should we formulate the cubic spline explicitly, by setting the $C$ as decision variables? Why or why not?")

# ╔═╡ d114264c-493d-4487-872f-5dfca77c378b
Foldable(md"Answer...", md"
In fact, we don't have to explicitly write out the polynomial coefficients $C$:
- We don't have to write out $C$ explicitly if $x[k]$ is already a decision variable (see below).
- If we remove $x[k]$ from the decision variables and represent $x[k]$ in terms of $C$, more variables ($C$) will be included in the (nonlinear) dynamics expressions, possibly making solving more challenging.
")

# ╔═╡ ccac876f-d7ce-416c-ac6d-0d657cbf82e1
md"
Here's how to formulate the dynamics constraints at the collocation points without explicitly using $C$ [^cmu13].

For each interval $(t_{k}, t_{k+1})$, the corresponding cubic function (shifted to $(0, t_{k+1} - t_{k}) := (0, h)$) satisfies

```math
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
1 & h & h^{2} & h^{3} \\
0 & 1 & 2h & 3h^{2}
\end{pmatrix}
\begin{pmatrix}
C_{0} \\ C_{1} \\ C_{2} \\ C_{3}
\end{pmatrix}
=
\begin{pmatrix}
x[k] \\ \dot{x}[k] \\ x[k+1] \\ \dot{x}[k+1]
\end{pmatrix}
```

```math
\implies
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
-\frac{3}{h^{2}} & -\frac{2}{h} & \frac{3}{h^{2}} & -\frac{1}{h} \\
\frac{2}{h^{3}} & \frac{1}{h^{2}} & -\frac{2}{h^{3}} & \frac{1}{h^{2}}
\end{pmatrix}
\begin{pmatrix}
x[k] \\ \dot{x}[k] \\ x[k+1] \\ \dot{x}[k+1]
\end{pmatrix}
=
\begin{pmatrix}
C_{0} \\ C_{1} \\ C_{2} \\ C_{3}
\end{pmatrix}
```

Now we can represent the variables at the collocation point, by plugging in the expressions of the $C$ we just derived:
```math
\begin{align*}
x(t_{k + \frac{1}{2}}) &= x(t_{k} + \frac{h}{2}) \\
	&= \frac{1}{2} (x[k] + x[k+1]) + \frac{h}{8} (\dot{x}[k] - \dot{x}[k+1]) \\
	&= \frac{1}{2} (x[k] + x[k+1]) + \frac{h}{8} (f(x[k], u[k]) - f(x[k+1], u[k+1])) \\
\dot{x}(t_{k + \frac{1}{2}}) &= \dot{x}(t_{k} + \frac{h}{2}) \\
	&= -\frac{3}{2h} (x[k] - x[k+1]) - \frac{1}{4} (\dot{x}[k] + \dot{x}[k+1]) \\
	&= -\frac{3}{2h} (x[k] - x[k+1]) - \frac{1}{4} (f(x[k], u[k]) + f(x[k+1], u[k+1]))
\end{align*}
```
For $u$, we simply have
```math
u(t_{k + \frac{1}{2}}) = u(t_{k} + \frac{h}{2}) = \frac{1}{2} (u[k] + u[k+1])
```
since its trajectory is approximated with a linear spline.

Now we can plug these expressions into the dynamics constraint at the collocation point
```math
\dot{x}(t_{k + \frac{1}{2}}) = f(x(t_{k + \frac{1}{2}}), u(t_{k + \frac{1}{2}}))
```
"

# ╔═╡ d75262d5-24b0-47f3-9010-264c43fa72e5
Foldable(md"Compressed vs. separated form", md"
We can either express the values of $x$ and $u$ at $t_{k + \frac{1}{2}}$ with the expressions above, or we can set them as decision variables, and enforce the expressions above as equality constraints.
Doing the former results in *compressed form* and the latter *separated form*.
The compressed form tends to perform better with a large number of segments and the separated form a small number of segments [^kelly2017].
")

# ╔═╡ 1861c438-eb0f-45cd-ad6c-f9289c80dc61
md"
Another perspective of the Hermite-Simpson collocation is the following.
One can approximate integrals with Simpson's rule for integration:
```math
\int_{t_{0}}^{t_{f}} w(\tau) d\tau \approx \sum_{k=0}^{N-1} \frac{h_{k}}{6} (w_{k} + 4w_{k+\frac{1}{2}} + w_{k+1})
```

When applied to the following expression implied by the dynamics
```math
\int_{t_{k}}^{t_{k+1}} \dot{x}(\tau) d\tau = \int_{t_{k}}^{t_{k+1}} f(x(\tau), u(\tau)) d\tau
```
the approximation is
```math
x[k+1] - x[k] = \frac{1}{6} h_{k} (f_{k} + 4f_{k + \frac{1}{2}} + f_{k+1})
```
where the short-hand notations like $f_{k}$ represent $f(x(t_{k}), u(t_{k}))$, etc.

Rearranging the terms, we can get
```math
-\frac{3}{2 h_{k}} (x[k+1] - x[k]) - \frac{1}{4} (f_{k} + f_{k+1}) = f_{k + \frac{1}{2}}
```
Note that the left hand side is exactly the expression we obtained for $\dot{x}(t_{k+\frac{1}{2}})$ earlier.
Hence, we have arrived at the same constraint approximating the system dynamics as earlier.

Under this perspective, we can write the optimization problem as
```math
\begin{align*}
\min_{x[\cdot], u[\cdot]} \quad & \ell_f (x[N]) + \sum_{k=0}^{N-1} h_{k} \ell(x[k], u[k]) \\
\text{s.t.} \quad
    & x[k+1] - x[k] = \frac{1}{6} h_{k} (f_{k} + 4f_{k + \frac{1}{2}} + f_{k+1}), \quad \forall k \in \{1, 2, ..., N - 1\} \\
	& x[0] = x_0 \\
	& ...
\end{align*}
```

Recall that $f_{k + \frac{1}{2}} = f(x(t_{k + \frac{1}{2}}), u(t_{k + \frac{1}{2}}))$, where the values of $x(t_{k + \frac{1}{2}})$ and $u(t_{k + \frac{1}{2}})$ are computed with the expressions we showed earlier (cubic and linear polynomials, respectively).

"

# ╔═╡ a5b2f0eb-ca36-4f88-a7d8-797bfc7f9657
md"
#### Hermite-Simpson Collocation: Cart-Pole
![cart-pole](https://epubs.siam.org/cms/10.1137/16M1062569/asset/images/fig_8.png) [^kelly2017]

Consider the cart-pole system, which contains a cart and a simple stiff pendulum.
The pendulum is not actuated, and the goal is to swing the pendulum to the upright position (and hold) by moving the cart horizontally.
"

# ╔═╡ ab93620c-0ca6-4be6-b525-8bafabbfbb32
html"""
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/Bzq96V1yN5k?rel=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
"""

# ╔═╡ 2ef066c2-701b-45c8-9807-d0d5d0556876
md"
The system dynamics is described by
```math
\ddot{q}_{1} = \frac{l m_{2} \sin(q_{2}) \dot{q}_{2}^{2} + u + m_{2} g \cos(q_{2}) \sin(q_{2})}{m_{1} + m_{2} (1 - \cos^{2}(q_{2}))}
```

```math
\ddot{q}_{2} = - \frac{l m_{2} \cos(q_{2}) \sin(q_{2}) \dot{q}_{2}^{2} + u \cos(q_{2}) + (m_{1} + m_{2}) g \sin(q_{2})}{l m_{1} + l m_{2} (1 - \cos^{2}(q_{2}))}
```

```math
x = \begin{pmatrix} q_{1} \\ q_{2} \\ \dot{q_{1}} \\ \dot{q_{2}} \end{pmatrix}
```

We will use the loss function
```math
\int_{0}^{t_{f}} u(\tau)^{2} d\tau
```
We could discretize this loss function as
```math
\sum_{k=0}^{N} u_{k}^{2}
```
or apply Simpson's rule for integration:
```math
\sum_{k=0}^{N-1} \frac{h_{k}}{6} (u_{k}^{2} + 4u_{k+\frac{1}{2}}^{2} + u_{k+1}^{2})
```
This example is adapted from [^kelly2017].
"

# ╔═╡ 4dd63a8d-d7e6-457a-a3df-ae982e49ce34
begin
	m_1 = 1  # mass of cart
	m_2 = 0.3  # mass of pole
	l = 0.5  # pole length
	g = 9.81  # gravity constant
	u_max = 20  # max actuator force
	d_max = 2  # max travel distance on both sides
	d = 1  # terminal position
	T = 2
	N = 32  # number of break points 
	h = T / (N - 1)  # length of each time interval

	function f(x, u)
		q_2 = x[2]
		q_1_d = x[3]
		q_2_d = x[4]
		q_1_dd_numer = l * m_2 * sin(q_2) * q_2_d + u + m_2 * g * cos(q_2) * sin(q_2)
		q_1_dd_denom = m_1 + m_2 * (1 - cos(q_2) ^ 2)
		q_1_dd = q_1_dd_numer / q_1_dd_denom
		q_2_dd_numer = l * m_2 * cos(q_2) * sin(q_2) * q_2_d^2 + u * cos(q_2) + (m_1 + m_2) * g * sin(q_2)
		q_2_dd_denom = l * m_1 + l * m_2 * (1 - cos(q_2) ^ 2)
		q_2_dd = -q_2_dd_numer / q_2_dd_denom
		return [q_1_d, q_2_d, q_1_dd, q_2_dd]
	end
end;

# ╔═╡ 94fbd45e-a3eb-403a-97af-52e8c93a1518
begin
	cp_model = Model(Ipopt.Optimizer)

	# initial guess: uniform motion, no control
	@variable(cp_model, -d_max .<= q_1[i=1:N] .<= d_max, start=i/N * d)
	@variable(cp_model, q_2[i=1:N], start=i/N * pi)
	@variable(cp_model, d_q_1[i=1:N], start=0)
	@variable(cp_model, d_q_2[i=1:N], start=0)
	@variable(cp_model, -u_max .<= u[1:N] .<= u_max, start=0)

	# collocation points
	@variable(cp_model, -d_max .<= q_1_c[1:N-1] .<= d_max)
	@variable(cp_model, q_2_c[1:N-1])
	@variable(cp_model, d_q_1_c[1:N-1])
	@variable(cp_model, d_q_2_c[1:N-1])
	@variable(cp_model, -u_max .<= u_c[1:N-1] .<= u_max)

	@expression(cp_model, x[i=1:N], [q_1[i], q_2[i], d_q_1[i], d_q_2[i]])
	@expression(cp_model, x_c[i=1:N-1], [q_1_c[i], q_2_c[i], d_q_1_c[i], d_q_2_c[i]])
	@expression(cp_model, f_xu[i=1:N], f(x[i], u[i]))
	@expression(cp_model, f_xu_c[i=1:N-1], f(x_c[i], u_c[i]))
	# collocation state
	@constraint(
		cp_model,
		[i=1:N-1],
		x_c[i] .== 1/2 * (x[i] .+ x[i+1]) + h/8 * (f_xu[i] - f_xu[i+1])
	)
	@constraint(
		cp_model,
		[i=1:N-1],
		u_c[i] == (u[i] + u[i+1]) / 2
	)
	# collocation dynamics
	@constraint(
		cp_model,
		[i=1:N-1],
		-3/(2 * h) * (x[i] - x[i+1]) - 1/4 * (f_xu[i] + f_xu[i+1]) .== f_xu_c[i]
	)
	
	# boundary constraints
	@constraint(cp_model, q_1[1] == 0)
	@constraint(cp_model, q_2[1] == 0)
	@constraint(cp_model, d_q_1[1] == 0)
	@constraint(cp_model, d_q_2[1] == 0)
	@constraint(cp_model, q_1[N] == d)
	@constraint(cp_model, q_2[N] == pi)
	@constraint(cp_model, d_q_1[N] == 0)
	@constraint(cp_model, d_q_2[N] == 0)
	
	@objective(cp_model, Min, sum(h/6 * (u[i]^2 + 4 * u_c[i]^2 + u[i+1]^2) for i in 1:N-1))
	
	optimize!(cp_model)
end

# ╔═╡ 100f8df3-e89c-4800-ab58-1d3ba7d524ac
begin
	fig = Figure()

	# Create three subplots (axes) arranged in one row with three columns
	ax1 = Axis(fig[1, 1]; ylabel="position (m)")
	ax2 = Axis(fig[2, 1]; ylabel="angle (rad)")
	ax3 = Axis(fig[3, 1]; xlabel="time(s)", ylabel="force (N)")
	
	# Plot data on each axis
	scatter!(ax1, range(0, T, N), value.(cp_model[:q_1]))
	scatter!(ax2, range(0, T, N), value.(cp_model[:q_2]))
	scatter!(ax3, range(0, T, N), value.(cp_model[:u]))
	
	# Display the figure
	fig
end

# ╔═╡ c27dfea7-2fc6-46a7-894b-6596e5f65d5a
question_box(md"Note that we set an initial guess for the state and control varialbes (using the `start` keyword when defining the variables). Even though this initial guess of the trajectories is physically impossible, it does help with convergence. Try removing the initial guess. Do you observe any change in the number of iterations for the solver to converge?")

# ╔═╡ 3c0a82ca-0b49-43c6-af74-c95dcb98e160
question_box(md"What do you think are the pros and cons of direct collocation?")

# ╔═╡ a27e052e-3496-47af-b2ed-77f8ee274568
md"
![TORA](https://repository-images.githubusercontent.com/308838583/4eb4d480-662c-11eb-97b7-96d8d0a4897c)
[TORA.jl](https://github.com/JuliaRobotics/TORA.jl) is a Julia package that formulates optimal control problem using direct transcription.

Let us see an example of solving a trajectory optimization problem of a robot arm with TORA.jl. This example is taken from the TORA tutorial [^tora].
"

# ╔═╡ b277b3be-906c-4459-820c-9ceb079c7211
begin
	vis = Visualizer()
	open(vis)
end

# ╔═╡ d9010ab2-a7aa-42bd-8701-63a698a9e2ea
begin
	# load a robot
	robot = TORA.create_robot_kuka("iiwa14", vis)
end

# ╔═╡ aaddf979-7cb3-4bf6-b446-ea507573c0b4
begin
	# To create a new problem, we use the TORA.Problem constructor, which takes three arguments:
	# 1. a TORA.Robot,
	# 2. the number of knots we wish to use for representing the trajectory, and
	# 3. the time step duration between each pair of consecutive knots.
	const duration = 2.0  # in seconds
	const hz = 150
	dt = 1/150
	# Create the problem
	problem = TORA.Problem(robot, Int(hz * duration + 1), dt)
	# Constrain initial and final joint velocities to zero
	TORA.fix_joint_velocities!(problem, robot, 1, zeros(robot.n_v))
	TORA.fix_joint_velocities!(problem, robot, problem.num_knots, zeros(robot.n_v))
	# End-effector constraint
	# circular path with acceleration at first and deceleration near the end
	let CubicTimeScaling(Tf::Number, t::Number) = 3(t / Tf)^2 - 2(t / Tf)^3
	    for k = 1:2:problem.num_knots  # For every other knot
	        θ = CubicTimeScaling(problem.num_knots - 1, k - 1) * 2π
	        pos = [0.5, 0.2 * cos(θ), 0.8 + 0.2 * sin(θ)]
	        TORA.constrain_ee_position!(problem, k, pos)
	    end
	end
	# Initial guess: static configuration, no velocity or torque
	initial_q = [0, 0, 0, -π/2, 0, 0, 0]
	initial_qs = repeat(initial_q, 1, problem.num_knots)
	initial_vs = zeros(robot.n_v, problem.num_knots)
	initial_τs = zeros(robot.n_τ, problem.num_knots)
	initial_guess = [initial_qs; initial_vs; initial_τs]
	initial_guess = vec(initial_guess)
	# remove control at the last knot
	initial_guess = initial_guess[1:end - robot.n_τ]

	# Solve with Ipopt
	cpu_time, sol, solver_log = TORA.solve_with_ipopt(problem, robot, initial_guess=initial_guess)
end;

# ╔═╡ 159ea3a5-e5d7-488c-af6a-6a9fd2114f70
TORA.play_trajectory(vis, problem, robot, sol)

# ╔═╡ 92e51bd2-630b-4882-a2d7-8d49121afc21
TORA.plot_log(solver_log)

# ╔═╡ 9932b4dc-4b6e-4a81-8f14-dc71c4c597fc
md"
## Differential Dynamic Programming

##### Approximate dynamic programming
This is an approach to solving the often intractable dynamic programming problem. We have covered dynamic programming in Class 4.

Instead of computing the value function at each time step exactly in its entirety,
1. Simulate one particular trajectory
2. (Backward pass) Update value function approximations to match the simulated data as well as possible: 
```math
\hat{V}_{t}(x_{t}) \approx \ell(x_{t}, u_{t}) + \hat{V}_{t+1}(x_{t+1})
```
3. (Forward pass) Simulate another trajectory by recomputing $x$ and $u$ using the updated $\hat{V}$.
4. Repeat until convergence criteria are met.
---

When exact dynamics is used to simulate the trajectories, approximate DP can be stopped at any iteration, and the current solution would satisfy the dynamics.
Collocation, on the other hand, may not be able to provide a solution that satisfies the dynamics over the entire time horizon until the solver is close to achieving convergence.
"

# ╔═╡ 65269bed-858b-4aa6-b8fc-c631a5b5b429
md"
---
The general idea of differential dynamic programming is approximating the value function with the second-order Taylor approximation around a nominal trajectory and updating the trajectory little-by-little in every iteration.

In the following, time step subscripts and indices will be omitted for conciseness, unless the context involves terms corresponding to different time steps, in which case we will use $t$ to denote the time step.

Let us write the second-order Taylor expansion of the value function near $x$ at a particular time step as
```math
V(x + \Delta x) \approx V(x) + V_{x}^{\top}(\Delta x) + \frac{1}{2} \Delta x^{\top} V_{xx} \Delta x
```
For the last time step $t=N$, we have
```math
V_{x} = \nabla_{x} \ell_{f}(x), \qquad V_{xx} = \nabla^{2}_{xx} \ell_{f}(x)
```

In our case, the definition of the action-value function (Q-function) is:
```math
Q[t](x[t] + \Delta x[t], u[t] + \Delta u[t]) =
```
```math
\ell_{t}(x[t] + \Delta x[t], u[t] + \Delta u[t]) + V[t+1](f(x[t] + \Delta x[t], u[t] + \Delta u[t]))
```

The second-order Taylor expansion of the action-value function (Q-function, the cost of the current action in the current state plus the value function of the new state) near $x$ and $u$ is
```math
Q(x + \Delta x, u + \Delta u) \approx Q(x, u) + \begin{pmatrix} Q_{x} \\ Q_{u} \end{pmatrix}^{\top} \begin{pmatrix} \Delta x \\ \Delta u \end{pmatrix}
+ \frac{1}{2} \begin{pmatrix} \Delta x \\ \Delta u \end{pmatrix}^{\top}
\begin{pmatrix}
	Q_{xx} & Q_{xu} \\
	Q_{ux} & Q_{uu}
\end{pmatrix}
\begin{pmatrix} \Delta x \\ \Delta u \end{pmatrix}
```
where the gradient and Hessian terms of $Q[t]$ are expressed in terms of $V_{x}[t+1]$ and $V_{xx}[t+1]$ (as well as gradient and Hessian of $f$; recall the definition of the $Q$ function).

By definition,
```math
V(x + \Delta x) = \min_{\Delta u} Q(x + \Delta x, u + \Delta u)
```

The gradient of the approximation of $Q(x + \Delta x, u + \Delta u)$ with respect to $\Delta u$ is
```math
Q_{u} + Q_{uu} \Delta u + Q_{ux} \Delta x
```
Assuming convexity of $Q(x + \Delta x, u + \Delta u)$, set the gradient to be $0$ and obtain
```math
\Delta u^{*} = -Q_{uu}^{-1} Q_{u} - Q_{uu}^{-1} Q_{ux} \Delta x := k + K \Delta x
```

As mentioned earlier, $k[t]$ and $K[t]$ depend on $V[t+1]$.
This implies that $V_{x}$ and $V_{xx}$ of each time step should be iteratively updated, starting from the last time step backward to the first time step.
So let us assume that we have updated $V[t+1]$, and would like to now use the updated $\Delta u^{*}$ to update $V_{x}[t]$ and $V_{xx}[t]$.

Plugging $\Delta u^{*}$ into $Q(x + \Delta x, u + \Delta u)$, we get $V(x + \Delta x)$ (since $\Delta u^{*}$ is a minimizer).
With some computation, we get the updated values
```math
V_{x} = Q_{x} + Q_{xu}^{\top} k \qquad V_{xx} = Q_{xx} + Q_{xu}^{\top} K
```

---

Overall, the algorithm can be summarized as:
1. Given the nominal trajectory $(x, u)$, compute the nominal values of $V(x)$.
2. Backward pass: from the last time step to the first, iteratively compute $\Delta u^{*}$ and update $V_{x}$ and $V_{xx}$.
3. Forward pass: recompute $\Delta x$ given the new $\Delta u$.
4. Repeat backward pass & forward pass until convergence criteria are met.
"

# ╔═╡ 71322a24-2eb6-48ef-b652-bd7105ccdea8
question_box(md"Can you think of one advantage collocation has over differential DP? (Hint: think about what is easy to be added to an optimization problem but not to the backward pass of differential DP).")

# ╔═╡ 98a56727-c565-4359-8c9d-73f2566e3413
Foldable(md"Answer...", md"
Additional constraints (on $x$ or $u$) can be easily imposed in collocation, but not so in differential DP. Techniques for imposing constraints in differential DP have been developed in the recent years, and you may read about them if interested.
")

# ╔═╡ Cell order:
# ╟─2fe513ba-9310-11f0-2266-9730fc13e5da
# ╟─e6aa5227-91bd-4cec-9448-24384708a305
# ╟─73e305d8-6619-4226-81cd-ee7a2fb59c3d
# ╟─d0c88449-5cb8-4cb2-9223-aae0482d240f
# ╟─27b4a8c3-bf81-4df7-a640-5037e947595f
# ╟─7899bee5-fdaf-4929-8073-c36d16c2e24f
# ╟─bfc7cced-3ce7-4f2b-8ee9-424d6d5ba682
# ╟─5f190a4e-d4b6-4279-8757-b0ec89df987f
# ╟─0de87fc1-7764-4194-8279-7c86fb3c64b8
# ╟─f49e647c-529d-4615-8bd2-7e18a7521eea
# ╟─dd00db44-8b37-44d6-9ace-0411c62e1dad
# ╟─9876a377-0da0-44a6-8922-378639724199
# ╟─626d09d9-4f46-427f-9b91-a95ba58a90da
# ╟─e3efd893-2d8f-42ae-93f4-85a5997ceee8
# ╟─8af4baab-a665-4111-b93f-d85ef8293ae8
# ╟─4b470369-ef0b-49bc-b013-05235eb28230
# ╟─1932a4b1-36fd-4f26-9a00-f7ea609791ae
# ╟─055ac28a-2fbd-4777-b683-688ae6b10a89
# ╟─a728d0f1-6c81-4342-9628-6957acbfda73
# ╟─359ae83f-af31-4363-bb35-19ec33586eae
# ╟─02589859-8f35-4b08-86f5-59698c6a6224
# ╟─165297a6-854f-475c-a16a-637de6dc9b69
# ╟─a193f3b5-5814-4528-9b07-92cdd493bd05
# ╟─d114264c-493d-4487-872f-5dfca77c378b
# ╟─ccac876f-d7ce-416c-ac6d-0d657cbf82e1
# ╟─d75262d5-24b0-47f3-9010-264c43fa72e5
# ╟─1861c438-eb0f-45cd-ad6c-f9289c80dc61
# ╟─a5b2f0eb-ca36-4f88-a7d8-797bfc7f9657
# ╟─ab93620c-0ca6-4be6-b525-8bafabbfbb32
# ╟─2ef066c2-701b-45c8-9807-d0d5d0556876
# ╠═e9f70acf-cadd-43a9-9068-0931e24f595c
# ╠═4dd63a8d-d7e6-457a-a3df-ae982e49ce34
# ╠═94fbd45e-a3eb-403a-97af-52e8c93a1518
# ╠═100f8df3-e89c-4800-ab58-1d3ba7d524ac
# ╟─c27dfea7-2fc6-46a7-894b-6596e5f65d5a
# ╟─3c0a82ca-0b49-43c6-af74-c95dcb98e160
# ╟─a27e052e-3496-47af-b2ed-77f8ee274568
# ╠═4de1acea-66ba-4e82-9426-8f3e152c3998
# ╠═b277b3be-906c-4459-820c-9ceb079c7211
# ╠═d9010ab2-a7aa-42bd-8701-63a698a9e2ea
# ╠═aaddf979-7cb3-4bf6-b446-ea507573c0b4
# ╠═159ea3a5-e5d7-488c-af6a-6a9fd2114f70
# ╠═92e51bd2-630b-4882-a2d7-8d49121afc21
# ╟─9932b4dc-4b6e-4a81-8f14-dc71c4c597fc
# ╟─65269bed-858b-4aa6-b8fc-c631a5b5b429
# ╟─71322a24-2eb6-48ef-b652-bd7105ccdea8
# ╟─98a56727-c565-4359-8c9d-73f2566e3413
