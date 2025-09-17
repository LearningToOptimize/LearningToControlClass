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

[^li2024]: F. Li, A. Abuduweili, Y. Sun, R. Chen, W. Zhao and C. Liu, “Continual Learning and Lifting of Koopman Dynamics for Linear Control of Legged Robots,” arXiv preprint arXiv:2411.14321, Nov. 2024.
	
[^mit1]: R. Tedrake, Ch. 1: Fully-actuated vs Underactuated Systems, Underactuated Robotics, MIT, 2024. [Online]. Available: [https://underactuated.mit.edu/intro.html](https://underactuated.mit.edu/intro.html)
		
[^mit10]: R. Tedrake, Ch. 10: Trajectory Optimization, Underactuated Robotics, MIT, 2024. [Online]. Available: [https://underactuated.mit.edu/trajopt.html](https://underactuated.mit.edu/trajopt.html)
"""
)
end

# ╔═╡ bfc7cced-3ce7-4f2b-8ee9-424d6d5ba682
md"
Trajectory optimization problems of systems with linear dynamics can likely be modeled as LQR (refer to Lecture 3), since quadratic functions are often good enough to be used as the cost.
Many nice properties then ensue.

However, the reality is often harsh.

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
Nonlinear dynamics causes the problem to be nonconvex. Nonlinearity could also arise from additional constraints.
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
![unicycle](https://media.licdn.com/dms/image/v2/D4E22AQHjrdv5VJpqYw/feedshare-shrink_1280/B4EZbV9cOjGkAo-/0/1747346378871?e=1758758400&v=beta&t=i1gPZeObZyh-tJurAUPvYcpW-El8fx8oIMoG51G9oxw)

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
For legged robots, a constraint of the form
```math
||f||_{2} \leq \mu f_{N}
```
may be enforced to prevent slipping, where $f$ is the frictional force, $\mu$ is the coefficient of statis friction and $f_{N}$ is the normal force.
"

# ╔═╡ 055ac28a-2fbd-4777-b683-688ae6b10a89
Foldable(md"Model choice: when does a linear controller suffice in robotics?", md"
> If you're not pushing the performance limit (e.g. of the actuators), then you can probably use a linear model. [^cmu11]

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
"

# ╔═╡ 02589859-8f35-4b08-86f5-59698c6a6224
md"
#### Direct Collocation
A special case of direct transcription which **approximates the $x$ and $u$ trajectories with splines (piecewise polynomials)**.

Discretize by setting **break points** in time which the splines are fit on.

A common choice: using a cubic spline for $x$ and a linear spline for $u$ (this is called Hermite-Simpson).


In this method, we would like to use a cubic spline to model the $x$ trajectory, so we will not enforce something like explicit Euler integration $x[n+1] = x[n] + h_{n} f(x[n], u[n])$ which would be a linear approximation.

Instead, we enforce a dynamics constraint at a point within every time interval,a **collocation point $t^{c}_{n}$**, often taken to be
```math
t^{c}_{n} = \frac{1}{2} (t_{n} + t_{n+1})
```
This makes sure that in each time interval $(t_{n}, t_{n+1})$, the approximate trajectory of $x$ is roughly consistent with the real dynamics.

![collocation](https://underactuated.mit.edu/data/collocation.svg)

The constraints will just be
```math
\dot{x}(t^{c}_{n}) = f(x(t^{c}_{n}), u(t^{c}_{n}))
```

The optimization problem is
```math
\begin{align*}
\min_{x[\cdot], u[\cdot]} \quad & \ell_f (x[N]) + \sum_{n=0}^{N-1} h_{n} \ell(x[n], u[n]) \\
\text{s.t.} \quad
    & \dot{x}(t^{c}_{n}) = f(x(t^{c}_{n}), u(t^{c}_{n})), \quad \forall n \in \{1, 2, ..., N - 1\} \\
	& x[0] = x_0 \\
	& ...
\end{align*}
```

Do we need to include $\dot{x}(t^{c}_{n})$, $x(t^{c}_{n})$, and $u(t^{c}_{n})$ all as decision variables? Let's find out.
"

# ╔═╡ 165297a6-854f-475c-a16a-637de6dc9b69
question_box(md"Why do you think one would choose cubic spline for $x$ and linear spline for $u$?")

# ╔═╡ a193f3b5-5814-4528-9b07-92cdd493bd05
question_box(md"If we have the explicit representation of the cubic spline $x(t) = C_{0} + C_{1}t + C_{2} t^{2} + C_{3} t^{3}$, then $x(t^{c}_{n})$ can be expressed in terms of the polynomial coefficients $C$. Should we formulate the cubic spline explicitly, by setting the $C$ as decision variables? Why or why not?")

# ╔═╡ d114264c-493d-4487-872f-5dfca77c378b
Foldable(md"Answer...", md"
In fact, we don't have to explicitly write out the polynomial coefficients $C$:
- We don't have to write out $C$ explicitly if $x[n]$ is already a decision variable (see below).
- If we remove $x[n]$ from the decision variables and represent $x[n]$ in terms of $C$, more variables ($C$) will be included in the (nonlinear) dynamics expressions, which may make solving more challenging.
")

# ╔═╡ ccac876f-d7ce-416c-ac6d-0d657cbf82e1
md"
Here's how to formulate the dynamics constraints at the collocation points using just $x$ and $u$ [^cmu13].

For each interval $(t_{n}, t_{n+1})$, the corresponding cubic function (shifted to $(0, t_{n+1} - t_{n}) := (0, h)$) satisfies

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
x[n] \\ \dot{x}[n] \\ x[n+1] \\ \dot{x}[n+1]
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
x[n] \\ \dot{x}[n] \\ x[n+1] \\ \dot{x}[n+1]
\end{pmatrix}
=
\begin{pmatrix}
C_{0} \\ C_{1} \\ C_{2} \\ C_{3}
\end{pmatrix}
```

Now we can represent the values at the collocation point:
```math
\begin{align*}
x(t^{c}_{n}) &= x(t_{n} + \frac{h}{2}) \\
	&= \frac{1}{2} (x[n] + x[n+1]) + \frac{h}{8} (\dot{x}[n] - \dot{x}[n+1]) \\
	&= \frac{1}{2} (x[n] + x[n+1]) + \frac{h}{8} (f(x[n], u[n]) - f(x[n+1], u[n+1])) \\
\dot{x}(t^{c}_{n}) &= \dot{x}(t_{n} + \frac{h}{2}) \\
	&= -\frac{3}{2h} (x[n] - x[n+1]) - \frac{1}{4} (\dot{x}[n] - \dot{x}[n+1]) \\
	&= -\frac{3}{2h} (x[n] - x[n+1]) - \frac{1}{4} (f(x[n], u[n]) - f(x[n+1], u[n+1]))
\end{align*}
```
For $u$, we simply have
```math
u(t^{c}_n) = u(t_{n} + \frac{h}{2}) = \frac{1}{2} (u[n] + u[n+1])
```
since its trajectory is approximated with a linear spline.

And we can replace the expressions into the dynamics constraint at the collocation point
```math
\dot{x}(t^{c}_{n}) = f(x(t^{c}_{n}), u(t^{c}_n))
```

(Note that all these are specific to one particular interval $(t_{n}, t_{n+1})$, even though the time indices are omitted for most of the notations.)

"

# ╔═╡ e9f70acf-cadd-43a9-9068-0931e24f595c
# (low-level) collocation problem formulation for simple system (hopefully)

# ╔═╡ 3c0a82ca-0b49-43c6-af74-c95dcb98e160
question_box(md"What do you think are the pros and cons of direct collocation?")

# ╔═╡ 4de1acea-66ba-4e82-9426-8f3e152c3998
# (high-level) TORA example

# ╔═╡ 9932b4dc-4b6e-4a81-8f14-dc71c4c597fc
md"
## Differential Dynamic Programming

##### Approximate dynamic programming
Instead of computing the value function at each time step exactly in its entirety,
1. Simulate one particular trajectory
2. (Backward pass) Update value function approximations to match the simulated data as well as possible: 
```math
\hat{V}_{t}(x_{t}) \approx \ell(x_{t}, u_{t}) + \hat{V}_{t+1}(x_{t+1})
```
3. (Forward pass) Simulate another trajectory by recomputing $u$ using the updated $\hat{V}$.
4. Repeat until convergence criteria are met.
---


"

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
# ╠═626d09d9-4f46-427f-9b91-a95ba58a90da
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
# ╠═e9f70acf-cadd-43a9-9068-0931e24f595c
# ╟─3c0a82ca-0b49-43c6-af74-c95dcb98e160
# ╠═4de1acea-66ba-4e82-9426-8f3e152c3998
# ╠═9932b4dc-4b6e-4a81-8f14-dc71c4c597fc
