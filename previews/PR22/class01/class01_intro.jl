### A Pluto.jl notebook ###
# v0.20.15

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
	class_dir = @__DIR__
	import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	# Pkg.status()
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
	import Images: load
end

# ╔═╡ b6ba1231-2942-4f06-8252-22f02553bb57
using CairoMakie

# ╔═╡ 29df2037-456f-4f98-9e32-71037e3d76c4
using ForwardDiff

# ╔═╡ ec473e69-d5ec-4d6a-b868-b89dadb85705
ChooseDisplayMode()

# ╔═╡ 8d7a34ef-5a2d-41a8-ac55-39ab00d7e432
md"
| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Rosemberg, Andrew |
|  Date   | : | 28 of July, 2025 |
"

# ╔═╡ ced1b968-3ba6-4e58-9bcd-bbc6bee2b93c
md"#### Reference Material"

# ╔═╡ 97994ed8-5606-46ef-bd30-c5343c1d99cf
begin
	MarkdownLiteral.@markdown(
"""

[^cmu]: Zachary Manchester et al. [Optimal Control and Reinforcement Learning at Carnegie Mellon University - CMU 16-745](https://optimalcontrol.ri.cmu.edu/)

[^OptProx]: Van Hentenryck, P., 2024. [Fusing Artificial Intelligence and Optimization with Trustworthy Optimization Proxies](https://www.siam.org/publications/siam-news/articles/fusing-artificial-intelligence-and-optimization-with-trustworthy-optimization-proxies/). Collections, 57(02).
		
[^ArmManip]: Guechi, E.H., Bouzoualegh, S., Zennir, Y. and Blažič, S., 2018. [MPC control and LQ optimal control of a two-link robot arm: A comparative study](https://www.mdpi.com/2075-1702/6/3/37). Machines, 6(3), p.37.

[^ZachMIT]: Zachary Manchester talk at MIT - [MIT Robotics - Zac Manchester - Composable Optimization for Robotic Motion Planning and Control](https://www.youtube.com/watch?v=eSleutHuc0w&ab_channel=MITRobotics).

[^Hespanha]: Hespanha, J.P., 2018. Linear systems theory. Princeton university press.
		
"""
)
end

# ╔═╡ 1f774f46-d57d-4668-8204-dc83d50d8c94
md"# Intro - Optimal Control and Learning

In this course, we are interested in problems with the following structure:

```math
\begin{equation}
\!\!\!\!\!\!\!\!\min_{\substack{(\mathbf u_1,\mathbf x_1)\\\mathrm{s.t.}}}
\!\underset{%
   \phantom{\substack{(\mathbf u_1,\mathbf x_1)\\\mathrm{s.t.}}}%
   \!\!\!\!\!\!\!\!\!\!(\mathbf u_1,\mathbf x_1)\in\mathcal X_1(\mathbf x_0)%
}{%
   \!\!\!\!c(\mathbf x_1,\mathbf y_1)%
}
+\mathbb{E}_1\Bigl[
   \quad \cdots
  
  \;+\;\mathbb{E}_t\Bigl[
    \min_{\substack{(\mathbf u_t,\mathbf x_t)\\\mathrm{s.t.}}}
    \!\underset{%
       \phantom{\substack{(\mathbf u_t,\mathbf x_t)\\\mathrm{s.t.}}}%
       \!\!\!\!(\mathbf u_t,\mathbf x_t)\in\mathcal X_t(\mathbf x_{t-1},w_t)%
    }{%
       \!\!\!\!\!\!\!\!\!\!c(\mathbf x_t,\mathbf u_t)%
    }
    +\mathbb{E}_{t+1}[\cdots]
\Bigr].
\end{equation}
```
which minimizes a first stage cost function $c(\mathbf{x}_1,
\mathbf{u}_1)$ and the expected value of future costs over possible
values of the exogenous stochastic variable $\{w_{t}\}_{t=2}^{T} \in
\Omega$. 

Here, $\mathbf{x}_0$ is the initial system state and the
control decisions $\mathbf{u}_t$ are obtained at every period $t$
under a feasible region defined by the incoming state
$\mathbf{x}_{t-1}$ and the realized uncertainty $w_t$. $\mathbf{E}_t$ represents the expected value over future uncertainties $\{w_{\tau}\}_{\tau=t}^{T}$. This
optimization program assumes that the system is entirely defined by
the incoming state, a common modeling choice in many frameworks (e.g.,
MDPs). This is without loss of generality,
since any information can be appended in the state. The system
constraints can be generally posed as:

```math
\begin{align}
    &\mathcal{X}_t(\mathbf{x}_{t-1}, w_t)= 
    \begin{cases}
        f(\mathbf{x}_{t-1}, w_t, \mathbf{u}_t) = \mathbf{x}_t \\
        h(\mathbf{x}_t, \mathbf{y}_t) \geq 0 
    \end{cases}
\end{align}
```
"

# ╔═╡ a0f71960-c97c-40d1-8f78-4b1860d2e0a2
md"""
where the outgoing state of the system $\mathbf{x}_t$ is a
transformation based on the incoming state, the realized uncertainty,
and the control variables. In the Markov Decision Process (MDP) framework, we refer to $f$ as the "transition kernel" of the system. State and
control variables are restricted further by additional constraints
captured by $h(\mathbf{x}_t, \mathbf{y}_t) \geq 0$.  We
consider policies that map the past information into decisions: $\pi_t : (\mathbf{x}_{t-1}, w_t) \rightarrow \mathbf{x}_t$. In
period $t$, an optimal policy is given by the solution of the dynamic
equations:

```math
\begin{align}
    V_{t}(\mathbf{x}_{t-1}, w_t) = &\min_{\mathbf{x}_t, \mathbf{u}_t} \quad  \! \! c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{E}_{t+1}[V_{t+1}(\mathbf{x}_t, w_{t+1})]    \\
    &   \text{ s.t. } \quad\mathbf{x}_t  = f(\mathbf{x}_{t-1}, w_t, \mathbf{u}_t) \nonumber         \\
    &  \quad \quad \quad \! \! h(\mathbf{x}_t, \mathbf{u}_t)  \geq 0. \nonumber             
\end{align}
```
```math
\implies \pi_t^{*}(\mathbf{x}_{t-1}, w_t) \in \{\mathbf{x}_t \;|\; \exists u_t \;:\; c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{E}_{t+1}[V_{t+1}(\mathbf{x}_t, w_{t+1})] = V_{t}(\mathbf{x}_{t-1}, w_t) \}
```

"""

# ╔═╡ 1d7092cd-0044-4d38-962a-ce3214c48c24
md"""
Function $V_{t}(\mathbf{x}_{t-1}, w_t)$ is refered to as the value function. To find the optimal policy for the $1^{\text{st}}$ stage, we need to find the optimal policy for the entire horizon $\{t=2,\cdots,T\}$ or at least estimate the "optimal" value function.
"""

# ╔═╡ 60ba261a-f2eb-4b45-ad6d-b6042926ccab
load(joinpath(class_dir, "indecision_tree.png"))

# ╔═╡ 15709f7b-943e-4190-8f40-0cfdb8772183
md"""
Notice that the number of "nodes" to be evaluated (either decisions or their cost) grows exponetially with the number of stages. This the the *Curse of dimensionality*
in stochastic programming.

"""

# ╔═╡ 5d7a4408-21ff-41ec-b004-4b0a9f04bb4f
question_box(md"Can you name a few ways to try and/or solve this problem?")

# ╔═╡ 7e487ebc-8327-4f3e-a8ca-1e07fb39991a
md"""
### Solution Methods

There are a few ways to solve these problems:

```math
(\mathbf{x}_{t-1}, w_t)\xrightarrow[\pi_t^{*}(\mathbf{x}_{t-1}, w_t)]{
\begin{align}
    &\min_{\mathbf{x}_t, \mathbf{u}_t} \quad  \! \! c(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{E}_{t+1}[V_{t+1}(\mathbf{x}_t, w_{t+1})]    \\
    &   \text{ s.t. } \quad\mathbf{x}_t  = f(\mathbf{x}_{t-1}, w_t, \mathbf{u}_t) \nonumber         \\
    &  \quad \quad \quad \! \! h(\mathbf{x}_t, \mathbf{u}_t)  \geq 0. \nonumber             
\end{align}
} (\mathbf{x}_t^{*}, \mathbf{u}_t^{*}) 
```

**Exact Methods:**
 - Deterministic Equivalent: Explicitly model all decisions of all possible scenarios. (Good Luck!)
 - Stochastic Dual Dynamic Programming, Progressive Hedging, ... (Hard but doable for some class of problems.)

**Approximate Methods**: 
 - Approximate Dynamic Programming, (model-free and model-based)Reinforcement Learning, Two-Stage Decision Rules, ...
 - **Optimization Proxies**:

```math
\theta^{\star}
\;=\;
\operatorname*{arg\,min}_{\theta \in \Theta}
\;
\mathbb{E}\Bigl[\bigl\|\,\pi_t^{\ast}-\pi_t(\,\cdot\,;\theta)\bigr\|_{\mathcal F}\Bigr],
```

"""

# ╔═╡ bd623016-24ce-4c10-acb3-b2b80d4facc8
md"[^OptProx]"

# ╔═╡ 2d211386-675a-4223-b4ca-124edd375958
@htl """

<img src="https://www.siam.org/media/k2hls5wb/figure1.jpg">

"""

# ╔═╡ 45275d44-e268-43cb-8156-feecd916a6da
# ╠═╡ skip_as_script = true
#=╠═╡
Foldable(md"#### LearningToOptimize Project", @htl """
<div style="
  border:1px solid #ccc;
  border-radius:6px;
  padding:1rem;
  font-size:0.9rem;
  max-width:760px;
  line-height:1.45;
">

  <!-- ─────────────────────── header ─────────────────────── -->
  <h2 style="margin-top:0">LearningToOptimize&nbsp;Organization</h2>

  <p>
    <strong>LearningToOptimize&nbsp;(L2O)</strong> is a collection of open-source tools
    focused on the emerging paradigm of <em>amortized optimization</em>—using machine-learning
    methods to accelerate traditional constrained-optimization solvers.
    <em>L2O is a work-in-progress; existing functionality is considered experimental and may
    change.</em>
  </p>

  <!-- ─────────────────── repositories table ──────────────── -->
  <h3>Open-Source&nbsp;Repositories</h3>

  <table style="border-collapse:collapse;width:100%">
    <tbody>
      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/LearningToOptimize.jl"
             target="_blank">LearningToOptimize.jl</a>
        </td>
        <td style="padding:4px 6px;">
          Flagship Julia package that wraps data generation, training loops and evaluation
          utilities for fitting surrogate models to parametric optimization problems.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/andrewrosemberg/DecisionRules.jl"
             target="_blank">DecisionRules.jl</a>
        </td>
        <td style="padding:4px 6px;">
          Build decision rules for multistage stochastic programs, as proposed in
          <a href="https://arxiv.org/pdf/2405.14973" target="_blank"><em>Efficiently
          Training Deep-Learning Parametric Policies using Lagrangian Duality</em></a>.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/L2OALM.jl"
             target="_blank">L2OALM.jl</a>
        </td>
        <td style="padding:4px 6px;">
          Implementation of the primal-dual learning method <strong>ALM</strong>,
          introduced in
          <a href="https://ojs.aaai.org/index.php/AAAI/article/view/25520" target="_blank">
          <em>Self-Supervised Primal-Dual Learning for Constrained Optimization</em></a>.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/L2ODLL.jl"
             target="_blank">L2ODLL.jl</a>
        </td>
        <td style="padding:4px 6px;">
          Implementation of the dual learning method <strong>DLL</strong>,
          proposed in
          <a href="https://neurips.cc/virtual/2024/poster/94146" target="_blank">
          <em>Dual Lagrangian Learning for Conic Optimization</em></a>.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/L2ODC3.jl"
             target="_blank">L2ODC3.jl</a>
        </td>
        <td style="padding:4px 6px;">
          Implementation of the primal learning method <strong>DC3</strong>, as described in
          <a href="https://openreview.net/forum?id=V1ZHVxJ6dSS" target="_blank">
          <em>DC3: A Learning Method for Optimization with Hard Constraints</em></a>.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/BatchNLPKernels.jl"
             target="_blank">BatchNLPKernels.jl</a>
        </td>
        <td style="padding:4px 6px;">
          GPU kernels that evaluate objectives, Jacobians and Hessians for
          <strong>batches</strong> of
          <a href="https://github.com/exanauts/ExaModels.jl" target="_blank">ExaModels</a>,
          useful when defining loss functions for large-batch ML predictions.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/BatchConeKernels.jl"
             target="_blank">BatchConeKernels.jl</a>
        </td>
        <td style="padding:4px 6px;">
          GPU kernels for batched cone operations (projections, distances, etc.),
          enabling advanced architectures such as repair layers.
        </td>
      </tr>

      <tr>
        <td style="padding:4px 6px;vertical-align:top;">
          <a href="https://github.com/LearningToOptimize/LearningToControlClass"
             target="_blank">LearningToControlClass</a>
        </td>
        <td style="padding:4px 6px;">
          Course repository for <em>Special Topics on Optimal Control &amp; Learning</em>
          (Fall 2025, Georgia Tech).
        </td>
      </tr>
    </tbody>
  </table>

  <!-- ─────────────── datasets and weights ──────────────── -->
  <h3 style="margin-top:1.25rem;">Open Datasets and Weights</h3>

  <p>
    The
    <a href="https://huggingface.co/LearningToOptimize" target="_blank">
    LearningToOptimize&nbsp;🤗 Hugging Face organization</a>
    hosts datasets and pre-trained weights that can be used with L2O packages.
  </p>

</div>
""")
  ╠═╡ =#

# ╔═╡ c08f511e-b91d-4d17-a286-96469c31568a
md"## Example: Robotic Arm Manipulation"

# ╔═╡ b3129bcb-c24a-4faa-a5cf-f69ce518ea87
begin
	load(joinpath(class_dir, "nlp_robot_arm.png"))
end

# ╔═╡ c1f43c8d-0616-4572-bb48-dbb71e40adda
md"""
[^ArmManip]

The tip of the second link is computed using the direct geometric model:

```math
p(\theta_{1},\theta_{2}) \;=\;
\begin{cases}
x = L_{1}\,\sin\theta_{1} \;+\; L_{2}\,\sin\!\bigl(\theta_{1}+\theta_{2}\bigr),\\[6pt]
y = L_{1}\,\cos\theta_{1} \;+\; L_{2}\,\cos\!\bigl(\theta_{1}+\theta_{2}\bigr).
\end{cases}
\tag{1}
```
"""

# ╔═╡ 57d896ca-221a-4cfc-b37a-be9898fac923
begin
md"""

**State**  
```math
  \mathbf{x}_t=\begin{bmatrix}\theta_{1,t}&\theta_{2,t}&\dot\theta_{1,t}&\dot\theta_{2,t}\end{bmatrix}^{\!\top}
```

**Control**  
```math
  \mathbf{u}_t=\boldsymbol\tau_t=\begin{bmatrix}\tau_{1,t}&\tau_{2,t}\end{bmatrix}^{\!\top}
```

**Dynamics** (Euler sample time Δt)  
```math
  \mathbf{x}_{t+1}=f_d(\mathbf{x}_t,\mathbf{u}_t)
  \;\;\equiv\;
  \begin{bmatrix}
  \boldsymbol\theta_t+\Delta t\,\dot{\boldsymbol\theta}_t\\[2pt]
  \dot{\boldsymbol\theta}_t+\Delta t\,\mathcal{M}^{-1}(\boldsymbol\theta_t)(B(\boldsymbol\theta_t)\boldsymbol\tau_t + F(w_t) - C(\boldsymbol\theta_t,\boldsymbol{\dot\theta})\bigr)
  \end{bmatrix}
```

**Stage cost** 

```math
c(\mathbf{x}_t,\mathbf{u}_t)=
\underbrace{\|p(\boldsymbol\theta_t)-p_{\text{target}}\|_2^{2}}_{\text{tracking}}
+\;\lambda_\tau\|\boldsymbol\tau_t\|_2^{2}\;,
\qquad \lambda_\tau>0 .
```

Terminal cost  
$V_T(\mathbf{x}_T)=\|p(\boldsymbol\theta_T)-p_{\text{target}}\|_2^{2}$.

**Constraints**

```math
h(\mathbf{x}_t,\mathbf{u}_t)\ge 0\;:\;
\begin{cases}
\theta_{\min}\le\boldsymbol\theta_t\le\theta_{\max} &\text{(joint limits)}\\
|\dot{\boldsymbol\theta}_t|\le\dot\theta_{\max} &\text{(velocity limits)}\\
|\boldsymbol\tau_t|\le\tau_{\max} &\text{(actuator limits)}
\end{cases}
```

"""
end

# ╔═╡ 52005382-177b-4a11-a914-49a5ffc412a3
section_outline(md"A Crash Course:",md" (Continuous-Time) Dynamics
")

# ╔═╡ 8ea866a6-de0f-4812-8f59-2aebec709243
md"

The general form for the Continuous-Time Dynamics of a smooth system:

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
function pendulum(θ_deg = 60; L = 4, fsize = (520, 450), _xlims=nothing, _ylims=(-5, 5))
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
    if !isnothing(_ylims)
	    ylims!(ax, _ylims)
    end

	if !isnothing(_xlims)
		xlims!(ax, _xlims)
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
m \cdot l^{2} \cdot \ddot{\theta} + m \cdot g \cdot l \cdot sin(\theta) = u
```
where
```math
\begin{cases}
m & \text{Mass} \\
l & \text{Length of the pole} \\
\theta & \text{Pole angular position} \\
g & \text{Gravity} \\
u & \text{Torque exerted at axis}
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
\frac{-g sin(\theta)}{l} + \frac{u}{ml^{2}}
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
md"## Control--Affine Systems

Non--linear Systems of the form:
```math
\dot{x} = \underbrace{f_{o}(x)}_{\text{drift}} +  \underbrace{B(x)}_{\text{input Jacobian}}u
```

 $\implies$ Non--linear in the state but affine in the input/control.

Control--Affine Systems are common in many mechanical systems.

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


> The equations were discovered in the 1750s by Swiss mathematician Leonhard Euler and Italian mathematician Joseph-Louis Lagrange.

""")

# ╔═╡ 5a691d10-44f7-4d44-a2c9-a7d4d720b7cc
begin
md"""
#### 🚀 Detour: The Principle of Least Action 🚀

In the calculus of variations and classical mechanics, the Euler–Lagrange equations are a system of second-order ordinary differential equations whose solutions are stationary points of the given action functional: 

```math
\mathcal{S}[q(\cdot)] \;=\;
\int_{t_0}^{t_f} L\!\bigl(q(t),\; \dot q(t)\bigr)\,dt,
```

In classical mechanics:
		 
```math
L = \underbrace{\frac{1}{2} v^{\top}M(q)v}_{\text{Kinematic Energy}} - \underbrace{U(q)}_{\text{Potential Energy}}
```

"""
end

# ╔═╡ f3d155c6-5384-481a-8373-582e753ea8d6
question_box(md"What can you say about $M(q)$? When do we have a problem inverting it?")

# ╔═╡ ee5c5e2e-e9f1-4f94-95c9-21d506281ae1
md"""
A curve ($q^\star(t)$) is physically realised iff it is a stationary
point of ($\mathcal{S}$) :

```math
\delta\mathcal{S}=0
\;\;\Longrightarrow\;\;
\frac{d}{dt}\!\bigl(\tfrac{\partial L}{\partial\dot q}\bigr)
- \frac{\partial L}{\partial q}=0
\quad\Longrightarrow\quad
M(q)\,\ddot q + C(q,\dot q)\,\dot q + \nabla U(q)=0 .
```

"""

# ╔═╡ b9aeab8a-f8ea-4310-8568-5d6bda0bb4d3
question_box(md"Can you derive the stationary condition?")

# ╔═╡ 30a013a8-c02e-4816-af0d-9280473c916b
md"""
In most cases:
```math
q^{*} \in \arg \min_{q}
\int_{t_0}^{t_f} L\!\bigl(q(t),\; \dot q(t)\bigr)\,dt,
```

Now, suppose the configuration must satisfy a *gap function*  
$\phi(q)\ge 0$ (e.g. **contact with the ground**, obstacle avoidance, joint
limits).  
The variational problem becomes

```math
q^{*} \;\in\;
\arg\!\min_{q(\cdot)}
\int_{t_0}^{t_f} L\!\bigl(q(t),\dot q(t)\bigr)\,dt
\quad\text{s.t.}\quad
\phi\!\bigl(q(t)\bigr)\;\ge 0 \;\;\;\forall\,t.
```

Let $(t_k = t_0 + k,\Delta t)$ with $(k=0,\dots,N)$ and
$(q_k \approx q(t_k))$.
Using the midpoint rule we approximate the action by

```math
S_N(q_{0:N})
\;=\;
\sum_{k=0}^{N-1}
L\!\Bigl(
      \tfrac12\bigl(q_k+q_{k+1}\bigr),\;
      \tfrac{q_{k+1}-q_k}{\Delta t}
\Bigr)\,\Delta t,
```

and obtain the finite‐dimensional problem

```math
\begin{aligned}
\min_{q_1,\dots,q_{N}}
& \; S_N(q_{0:N}) \\[4pt]
\text{s.t.}\;&\;
   \phi(q_{k+1}) \;\ge 0,
   \qquad k = 0,\dots,N-1.
\end{aligned}
```

"""

# ╔═╡ 2cc57795-717a-46f0-9bb5-67b601a766de
begin
	gif_url   = "https://raw.githubusercontent.com/dojo-sim/Dojo.jl/main/docs/src/assets/animations/atlas_drop.gif"
	still_url = "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQkrtL7TCGzNxFlXIqYHW_cFP9pfLscwd7vLSH09nfRFEQCqX_J"
	md""
end

# ╔═╡ 59f6167d-796c-4844-89c0-c796fb59aa2e
Columns(md"[^ZachMIT]", md"▶/⏸$(@bind playing CheckBox(default=false))")

# ╔═╡ 58c2e1f2-819d-40fc-8e92-03a1a3019a3d
Columns(md"""
$(load(joinpath(class_dir, "rocket_physics.png")))

#### Dojo.jl
		
A differentiable physics engine for robotics that simulates systems using optimization.

- [ArXiv preprint](https://arxiv.org/abs/2203.00806)
- [GitHub](https://github.com/dojo-sim/Dojo.jl)
		
"""
, 
@htl """
<img src="$(playing ? gif_url : still_url)"
	 width="800" height="600"
	 style="object-fit:contain;" />
"""		
)

# ╔═╡ 70690e72-c31e-4c91-b211-35c74d1d9973
warning_box(md"But in general we need a *ReFeynman* of the these equations!")

# ╔═╡ 5f35a169-887f-477f-b010-167627f7ce4c
md"## (State–Space) Linear Systems

A system is **Continuous Linear** (CLTV / CLTI) if it can be written as:

```math
\dot{x} = A_{t}x + B_{t}u
```

In state–space literature we often also see a **Output Equation**:
```math
y = C_{t}x + D_{t}u
```

but we will neglect it for now.
"

# ╔═╡ 5c8f6256-e818-4aa1-aea0-02422df8f77c
Foldable(md" When do we have a Time--Invariant (TI) vs Time--Variant (TV)?", md"""

When (A,B) are constant we have an LTI system; otherwise it is LTV.

""")

# ╔═╡ a3f47dad-3cfa-4f6d-9dc6-d4b09d209f86
md"
**Non--Linear Systems are often approximated by Linear Systems (locally).**
"

# ╔═╡ e860d92b-cc8f-479b-a0fc-e5f7a11ae1fd
Foldable(md" $\dot{x} = f(x,u) \; \implies \; A=? \; B=?$", md"""

Suppose now that we apply our dynamics equation to an input:

```math
u(t) = u_{eq} + \delta u(t), \quad t \ge 0
```
where $u_{eq}$ is an fixed input and $\delta u(t)$ is a perturbation function such that the input is close 
but not equal to $u_{eq}$ and similarly we perturb the initial condition:

```math
x(0) = x_e + \delta x(0)
```

We will define the deviation from the reference state as:
```math
\delta x(t) = x(t) - x_e, \quad t \ge 0
```

To determine the evolution of $\delta x(t)$, we can expand the dynamics around the reference point using a Taylor expansion:

```math
\dot{\delta x}(t) = f(x_e + \delta x(t), u_{eq} + \delta u(t))
```
```math
=\frac{\partial f}{\partial x}\bigg|_{(x_e, u_{eq})} \delta x(t) + \frac{\partial f}{\partial u}\bigg|_{(x_e, u_{eq})} \delta u(t) + \mathcal{O}(\|\delta x\|^2) + \mathcal{O}(\|\delta u\|^2)
```

Considering just the first-order terms we obtain:

```math
A= \frac{\partial f}{\partial x}|_{(x_e,u_e)}
, \quad B= \frac{\partial f}{\partial u}|_{(x_e,u_e)}
```

**Attention!** The linearization describes perturbations around the reference $(x_e,u_e)$; it is valid only while $\|\delta x\|$ and $\|\delta u\|$ remain small.

""")

# ╔═╡ bb4bfa72-bf69-41f5-b017-7cbf31653bae
Foldable(md"Why approximate? What happens to the optimal control problem?", md"""

The problem becomes convex!!

""")

# ╔═╡ 2936c97e-a407-4e56-952f-0a2dfb7acf83
md"""## Equilibria

A **Equilibrium** point $(x_{\mathrm{eq}},u_{\mathrm{eq}})$ is one at which the system is and will remain at "rest":

```math
\dot{x} = f(x_{\mathrm{eq}},u_{\mathrm{eq}}) = 0
```

The root of the dynamic equations!

In this case,

```math
x(t) = x_{\mathrm{eq}}, \; u(t) = u_{\mathrm{eq}} \; \forall t
``` 
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
\frac{-g sin(\theta)}{l} + \frac{u}{ml^{2}}
\end{bmatrix}=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\end{cases}
```
```math
\implies \frac{u}{ml^{2}} = \frac{g sin(\pi / 2)}{l}
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
	plt = plot(range(-2.2,2.2, 1000),f, label="ẋ = x³ - 3x", xlabel="x",
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

#### Lyapunov stability

Our System

 - State vector: $x(t)\in\mathcal D\subseteq\mathbb R^{n}$   
		 
 - Open set containing the origin: $\mathcal D$
		 
 - System is continuous: $f:\mathcal D\to\mathbb R^{n}$.

 - Assume an **equilibrium** $x_e$ exists, i.e. $f(x_e)=0$.

---

1. **Lyapunov stability**  
   The equilibrium is Lyapunov stable if, for every $\epsilon>0$, there exists $\delta>0$ such that  
```math
   \|x(0)-x_e\|<\delta \;\Longrightarrow\; \|x(t)-x_e\|<\epsilon\quad\forall\,t\ge 0.
```

2. **Asymptotic stability**  
   The equilibrium is asymptotically stable if it is Lyapunov stable **and** there exists $\delta>0$ such that  
```math
   \|x(0)-x_e\|<\delta \;\Longrightarrow\; \lim_{t\to\infty}\|x(t)-x_e\|=0.
```

3. **Exponential stability**  
   The equilibrium is exponentially stable if it is asymptotically stable **and** there exist $\alpha>0,\;\beta>0,\;\delta>0$ with  

```math
\|x(0)-x_e\|<\delta \;\Longrightarrow\;
\|x(t)-x_e\|\;\le\;\alpha\,\|x(0)-x_e\|\,e^{-\beta t},
\qquad t\ge 0.
```

---

Intuitive meaning:

* **Lyapunov stable** – Trajectories that start *sufficiently close* stay *arbitrarily close* for all future time.  
* **Asymptotically stable** – Those trajectories not only stay close but also converge to $x_e$.  
* **Exponentially stable** – Convergence is at least as fast as the known rate $\alpha\|x(0)-x_e\|e^{-\beta t}$.

""")

# ╔═╡ 161a2a6e-567f-4994-8d77-9a0f0962cdd9
md"""
As we increase the dimensions, it gets increasingly more complicated to reason about how a system will evolve!

> For continuous, autonomous dynamical systems, the Poincaré–Bendixson theorem states that chaos cannot occur in phase spaces of dimension less than 3.
"""

# ╔═╡ f08c95ef-f784-4c88-be61-91e0549d421b
md"### $\mathbb{R}^n Space$

Sensitivity is now defined by a Jacobian Matrix:

```math
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_{1}} & \dots & \frac{\partial f_1}{\partial x_{n}} \\
\vdots & \frac{\partial f_i}{\partial x_{j}} & \vdots \\
\frac{\partial f_n}{\partial x_{1}} & \dots & \frac{\partial f_n}{\partial x_{n}}
\end{bmatrix}
```


"

# ╔═╡ 41e1934a-2a43-44c3-9bcf-bef56f4d057e
Foldable(md"How to determine if Asymptotically Stable?", md"""

The Eigen-value decomposition $J = V \Lambda V^{-1}$ provides us with a decomposition into "n" 1--D Systems. Then:
		 
```math
\forall \lambda \; : \; \mathcal{R}(\lambda) < 0 \quad \implies \; \text{Asymptotically Stable}
```

""")

# ╔═╡ fc585231-a650-4efb-aea4-95110dbf8fa5
Foldable(md"Pendulum?", md"""

	 
```math
f(x) =
\begin{bmatrix}
\dot{\theta} \\
\frac{-g sin(\theta)}{l}
\end{bmatrix}
\implies
J=\begin{bmatrix}
0 & 1 \\
\frac{-g cos(\theta)}{l} & 0  
\end{bmatrix}
```
```math
\begin{cases}
\text{eigvals}(J_{\theta = \pi}) = \pm\sqrt{g/l} & \text{Unstable} \\
\text{eigvals}(J_{\theta = 0}) = \pm \; i \;\sqrt{g/l} & \text{Lyapunov Stable | Undampled Oscillation}\\
\end{cases}
```

But the $2^{\text{nd}}$ case is still not asymptotically stable!

""")

# ╔═╡ da8a1e40-7e7c-472a-933a-c585754270bd
question_box(md"Can we add a continuous controler to make it asymptotically stable?")

# ╔═╡ 211f75ed-8acb-4261-bf97-8fc70be2a79b
Columns(md"""
### Example Unicycle Model

Consider a unicycle moving in a plane, described by the continuous-time dynamics

```math
\dot p_x = v\cos\theta,\qquad
\dot p_y = v\sin\theta,\qquad
\dot\theta = \omega ,
```

where the control input is given by
$u = [v,\,\omega]^\top$ and $(p_x, p_y)$ are the Cartesian coordinates of the wheel and $\theta$ its orientation. 

#### Coordinate transformation

Introduce the new coordinates

```math
x = \begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}
:=
\begin{bmatrix}
p_x\cos\theta + (p_y-1)\sin\theta \\[4pt]
-\,p_x\sin\theta + (p_y-1)\cos\theta \\[4pt]
\theta
\end{bmatrix}
```

This transformation can be viewed as a rotation of the position vector about $\theta$.
""",@htl """

<img src="https://media.licdn.com/dms/image/v2/D4E22AQEL2gwfuHbjyg/feedshare-shrink_2048_1536/B4EZbV9cNyHMAo-/0/1747346377441?e=1758758400&v=beta&t=KsIL2efm-3OBWsn8rRMRdqVBDIk65og6WGYxMo7pwZc">

<img src="https://media.licdn.com/dms/image/v2/D4E22AQHjrdv5VJpqYw/feedshare-shrink_1280/B4EZbV9cOjGkAo-/0/1747346378871?e=1758758400&v=beta&t=i1gPZeObZyh-tJurAUPvYcpW-El8fx8oIMoG51G9oxw">

""")

# ╔═╡ 7e7ce801-a510-4ef2-910b-9e10b685ea58
md"""[Mini Wheelbot ICRA2025](https://www.linkedin.com/posts/sebastian-trimpe-2472a0a3_icra2025-ieee-robotics-activity-7328901914195103746-zr5j/)"""

# ╔═╡ 7ad76460-e60c-4579-bee0-ac6f3c511877
Foldable(md"#### Nonlinear state-space representation",
md"""
Differentiating each component gives

```math
\dot x_1 = v + \omega x_2, \qquad
\dot x_2 = -\omega x_1, \qquad
\dot x_3 = \omega.
```

Thus the system in compact form is

```math
\dot x =
\begin{bmatrix}
v + \omega x_2 \\[2pt]
-\omega x_1 \\[2pt]
\omega
\end{bmatrix}
```

This shows a control-affine nonlinear system.
"""		
)

# ╔═╡ e055fdc3-fb83-4c13-85c9-6edcb661e2dd
Foldable(md"#### Linearization around the equilibrium ($x^{\ast} = 0, \; u^{\ast} = 0$)",
md"""
Define perturbations $\delta x = x - x^{\ast}, \; \delta u = u - u^{\ast}$.
The linearized dynamics are obtained from the Jacobians

```math
A = \left.\frac{\partial f}{\partial x}\right|_{(x^{\ast},u^{\ast})}, 
\quad
B = \left.\frac{\partial f}{\partial u}\right|_{(x^{\ast},u^{\ast})}.
```

Evaluating,

```math
A = 0_{3\times 3},
\qquad
B =
\begin{bmatrix}
1 & 0 \\
0 & 0 \\
0 & 1
\end{bmatrix}.
```

"""		
)

# ╔═╡ db1c0c1e-9f3b-44b6-8974-6d5ab5e5e8b7
md"""
## Local Linearizations around Trajectories

Often it is convenient to consider perturbations around an arbitrary (feasible) reference trajectory (*ref*) ($x_{ref}(t)$, $u_{ref}(t)$) instead of an equilibrium point.

Now, assuming an input perturbed by a small signal $\delta u(t)$:
```math
u(t) = u_{ref}(t) + \delta u(t), \quad t \ge 0
```
and a deviation from the reference initial condition:
```math
x(0) = x_{ref}(0) + \delta x(0)
```
We can define the deviation from the reference state as:
```math
\delta x(t) = x(t) - x_{ref}(t), \quad t \ge 0
```
To determine the evolution of $\delta x(t)$, we can expand the dynamics around the reference point using a Taylor expansion:
```math
\dot{\delta x}(t) = f(x_{ref}(t) + \delta x(t), u_{ref}(t) + \delta u(t))
```
```math
=\frac{\partial f}{\partial x}\bigg|_{(x_{ref}(t), u_{ref}(t))} \delta x(t) + \frac{\partial f}{\partial u}\bigg|_{(x_{ref}(t), u_{ref}(t))} \delta u(t) + \mathcal{O}(\|\delta x\|^2) + \mathcal{O}(\|\delta u\|^2)
```
Considering just the first-order terms we obtain:
```math
A(t)= \frac{\partial f}{\partial x}|_{(x_{ref}(t),u_{ref}(t))}
, \quad B(t)= \frac{\partial f}{\partial u}|_{(x_{ref}(t),u_{ref}(t))}
```

 > In general, local linearizations around trajectories lead to LTV systems because the partial derivatives need to be computed along the trajectory. However, for some nonlinear systems there are trajectories for which local linearizations actually lead to LTI systems. For models of vehicles (cars, airplanes, helicopters, hovercraft, submarines, etc.) trajectories that lead to LTI local linearizations are called trimming trajectories. They often correspond to motion along straight lines, circumferences, or helices. [^Hespanha]

"""

# ╔═╡ 04b6560f-aee2-41fd-86fa-d075f0b3d738
md"""
### Example Unicycle Model

Consider the Reference motion
```math
\omega(t)=1,\qquad v(t)=1,\qquad
p_x(t)=\sin t,\qquad
p_y(t)=1-\cos t,\qquad
\theta(t)=t,\quad t\ge 0 .
```
Then
```math
\dot p_x(t)=\cos t = v\cos\theta,\qquad
\dot p_y(t)=\sin t = v\sin\theta,\qquad
\dot\theta(t)=1=\omega,
```
so this is indeed a solution of the system.

In the rotated coordinates,
```math
\begin{align}
x_1 &= p_x\cos\theta + (p_y-1)\sin\theta \\
    &= \sin t\cos t + (-\cos t)\sin t = 0, \\
x_2 &= -p_x\sin\theta + (p_y-1)\cos\theta \\
    &= -\sin^2 t - \cos^2 t = -1, \\
x_3 &=\theta=t.
\end{align}
```

Hence the reference trajectory in $x$--coordinates is
```math
x_{\mathrm{ref}}(t) = \begin{bmatrix}0\\-1\\t\end{bmatrix},\qquad
u_{\mathrm{ref}}(t)=\begin{bmatrix}1\\1\end{bmatrix}.
```
"""

# ╔═╡ c7b11f27-1582-45d0-adef-32eb9c6de588
Foldable(md"Linearization along the trajectory",
md"""
Let $f(x,u)=\big[v+\omega x_2,\ -\omega x_1,\ \omega\big]^\top$.
The Jacobians are
```math
A(x,u)=\frac{\partial f}{\partial x}=
\begin{bmatrix}
0&\omega&0\\
-\omega&0&0\\
0&0&0
\end{bmatrix},\qquad
B(x,u)=\frac{\partial f}{\partial u}=
\begin{bmatrix}
1&x_2\\
0&-x_1\\
0&1
\end{bmatrix}.
```
Evaluating on $(x_{\mathrm{ref}}(t),u_{\mathrm{ref}}(t))$ gives the constant matrices
```math
A=\begin{bmatrix}
0&1&0\\
-1&0&0\\
0&0&0
\end{bmatrix},\qquad
B=\begin{bmatrix}
1&-1\\
0&0\\
0&1
\end{bmatrix},
```
Therefore time-invariant (LTI) linearization!
""")

# ╔═╡ 4cd2306d-e3f3-4895-8798-596f6c353bdc
question_box(md"### How do we include the dynamics in control/decision problems?")

# ╔═╡ ca9d4d0c-40c2-4144-866f-db1417d42c8f
md"""
## Discrete--Time Dynamics

We mostly plan in descrete--time, so we need $x(t)$ for the horizon of our planning!

"""

# ╔═╡ 3a576353-76bb-4c12-b2a2-b37e8e1dd17f
Foldable(md"2 Main issues!",md"""
 - Often, there is no analytical solutions $x(t)$ for a general PDE!

 - PDEs can't describe discontinuous events!
""")

# ╔═╡ 871587c3-380a-4492-b680-aa7b6dd2004f
md"""### Explicit Form:

```math
x_{t+1} = f_d(x_t, u_t)
```

#### Foward Euler Integration:

```math
x_{t+1} = x_t + \underbrace{\Delta_t \cdot f(x_t, u_t)}_{f_d}
```

"""

# ╔═╡ 27b490fa-8c2e-4a1a-a6d6-d57ad149a066
"""
	pendulum_dynamics(x)

Defines the continuous dynamics of a pendulum. 

Expects `x = [θ, θ̇]` | 
returns `[θ̇; θ̈]`.
"""
function pendulum_dynamics(x)
	l = 1.0 # Length of the Pole
	g = 9.81 # Gravity
	
	θ = x[1] # Angle
	θ̇ = x[2] # Angular velocity
	
	θ̈ = -(g/l)*sin(θ) # ODE
	
	return [θ̇; θ̈] # Return ẋ
end

# ╔═╡ f6c075f9-9d79-46ba-870e-e12c2b3357df
"""
	pendulum_forward_euler(fun, x0, Tf, h) 

Foward Euler Integrator.

Arguments:
 - `fun`: Continuous Dynamics;
 - `x0`: Intitial Condition;
 - `Tf`: Final Time;
 - `h`: Time--Step duration;
"""
function pendulum_forward_euler(fun, x0, Tf, h)    
    t = Array(range(0,Tf,step=h)) # Time-Steps
    
    x_hist = zeros(length(x0),length(t)) # Visited States
    x_hist[:,1] .= x0 # Initial Condition
    
    for k = 1:(length(t)-1)
        x_hist[:,k+1] .= x_hist[:,k] + h*fun(x_hist[:,k]) # Transition
    end
    
    return x_hist, t
end

# ╔═╡ e541f564-a1f1-4fa9-a62c-86265722857a
Columns(pendulum(0.1; _xlims=(-0.02, 0.02)), question_box(md"What will happen?"))

# ╔═╡ 166138f8-4c74-426e-8ce6-2e9c4416de9f
md"""
 Sim: $(@bind sim1 CheckBox())
"""

# ╔═╡ facaecf5-2d27-4707-8087-0aa18517d7cd
begin
	if sim1
		local x0 = [.1; 0] # Starting point very close to the Lyapunov Stable point!
		x_hist1, t_hist1 = pendulum_forward_euler(pendulum_dynamics, x0, 5., .1)
		plot(t_hist1, x_hist1[1,:], xlabel="t", ylabel="x(t)", label="")
	end
end

# ╔═╡ 19f3d541-7452-4bcc-89e0-51bf5dab34e6
Foldable(md"#### Why?",md"""

### Forward-Euler update for the undamped pendulum  

```math
x_t=\begin{bmatrix}\theta_t\\\dot{\theta}_t\end{bmatrix},
\qquad
\dot{x}_t=
\begin{bmatrix}
\dot{\theta}_t\\
-\dfrac{g}{l}\,\sin\theta_t
\end{bmatrix}
```

Forward-Euler integration gives  

```math
x_{t+1}=x_t+\Delta t\,\dot{x}_t
\;\;\Longrightarrow\;\;
\begin{cases}
\theta_{t+1}= \theta_t + \Delta t\,\dot{\theta}_t\\[8pt]
\dot{\theta}_{t+1}= \dot{\theta}_t - \dfrac{g}{l}\,\Delta t\,\sin\theta_t
\end{cases}
```

---

## Mechanical energy  

```math
E(\theta,\dot{\theta}) \;=\;
\frac12\,m l^{2}\,\dot{\theta}^{2}
\;+\;mgl\bigl(1-\cos\theta\bigr).
```

Define $E_t = E(\theta_t,\dot{\theta}_t)$ and $E_{t+1}=E(\theta_{t+1},\dot{\theta}_{t+1})$.
Using the update rules and a second-order expansion of $\cos(\theta_{t+1})$,

```math
\cos(\theta_{t+1})
=\cos\bigl(\theta_t+\Delta t\,\dot{\theta}_t\bigr)
\approx
\cos\theta_t
-\Delta t\,\dot{\theta}_t\sin\theta_t
-\frac{(\Delta t\,\dot{\theta}_t)^2}{2}\cos\theta_t.
```

The resulting **energy increment** is  

```math
\Delta E := E_{t+1}-E_t
=
\frac{m(\Delta t)^2}{2}
\Bigl[g^{2}\sin^{2}\theta_t
\;+\;g\,l\,\dot{\theta}_t^{2}\cos\theta_t\Bigr]
+\mathcal O\!\bigl((\Delta t)^3\bigr).
```

For small-amplitude oscillations ($|\theta_t|<\pi/2 \;\rightarrow\; \cos\theta_t>0$),
both terms inside the bracket are non-negative, so  

```math
\boxed{\;\Delta E>0\quad\text{whenever }(\theta_t,\dot{\theta}_t)\neq(0,0).}
```

---

### Why Forward-Euler injects energy  

* The slope $-\tfrac{g}{l}\sin\theta_t$ is held **constant** over the whole step.  
* Because $|\sin\theta|$ increases with $|\theta|$ near the origin, this straight-line extrapolation **overestimates** the restoring torque.  
* Consequently $\dot{\theta}_{t+1}$ (and then $\theta_{t+1}$) overshoot, injecting a fixed energy surplus of order $(\Delta t)^2$ at every step.

This systematic surplus appears in any undamped oscillatory system; Forward-Euler trajectories spiral outward, while symplectic or energy-preserving schemes (e.g. semi-implicit Euler, Verlet) avoid the drift.	 
""")

# ╔═╡ 800341e2-de0d-43a4-b0f1-a74021963f43
md"""### Stability of Discrete--Time Systems

In discrete time, dynamics is defined by function composition:

```math
x_t = f_d(f_d(\dots f_d(x_0) \dots ))
```

The *linearized* derivative of this system is:

```math
\frac{\partial x_t}{\partial x_0} = \frac{\partial f_d}{\partial x}\Bigg|_{x_0}
\;\frac{\partial f_d}{\partial x}\Bigg|_{x_0} \dots \frac{\partial f_d}{\partial x}\Bigg|_{x_0}=J_d^t
```

Assume we are at a equilibrium and we have changed coordinates for $x_{0}=0$
"""

# ╔═╡ 7acd26bc-e35b-47a4-aca3-719f106f3238
Foldable(md"Conditions on $J_d$ for the system to be stable at $x_0$?",md"""

```math
\lim_{t \rightarrow \infty} J_d^{t} x_0 = 0 \quad \forall x_0
```
```math
\implies \lim_{t \rightarrow \infty} J_d^{t} = 0
```
```math
\implies |\text{eigvals}(J_d)| < 1
```	 
""")

# ╔═╡ c94a2f37-9782-4fd8-bae3-61fa8f82ca2d
md"""
#### pendulum\_euler\_Ad

The pendulum Forward--Euler integration:  

```math
x_{t+1}=x_t+ \underbrace{\Delta t\,f(x_t)}_{f_d(x_t)}
```

```math
\implies J_d = \frac{\partial f_d}{\partial x} = I + \Delta_t J = I + \Delta_t \begin{bmatrix}
0 & 1 \\
-\dfrac{g \cos(\theta)}{l} & 0
\end{bmatrix}
```

"""

# ╔═╡ 5a1cdae4-2b2f-4df0-866a-6e62be6ddb4a
function pendulum_euler_Ad(x0, h)
    g = 9.81 # Gravity
    Ad = [1 h; -g*h*cos(x0[1]) 1]
end

# ╔═╡ 8903c3cc-fc4b-4ffa-bf5b-7645724e8b02
eigvals(pendulum_euler_Ad(0, 0.001))

# ╔═╡ bc3b48bd-0a4b-4282-84dd-cbb7bf6b084e
begin
	eignorm = zeros(100)
	h = LinRange(0,0.1,100)
	for k = 1:length(eignorm)
	    eignorm[k] = max(norm.(eigvals(pendulum_euler_Ad([0;0], h[k])))...)
	end
	plot(h,eignorm, xlabel="Δₜ", ylabel=L"|Λ|_{∞}", label="")
end

# ╔═╡ 7ebc1af0-e8d7-40b2-8395-48aaacb272de
md"#### Always unstable!"

# ╔═╡ cc5a6c0f-bf72-4e5d-aed6-7cbbadac862a
md"""
#### 4ᵗʰ--Order Runge-Kutta Method

Fits a cubic polynomial to to $x(t)$. 

**A better explicit integrator than when fitting a line!**
"""

# ╔═╡ f722b8da-0440-4bc1-8124-84305ef4bd4d
function fd_pendulum_rk4(xk, h)
    f1 = pendulum_dynamics(xk)
    f2 = pendulum_dynamics(xk + 0.5*h*f1)
    f3 = pendulum_dynamics(xk + 0.5*h*f2)
    f4 = pendulum_dynamics(xk + h*f3)
    return xk + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

# ╔═╡ 22ab7266-894c-457c-ad19-1c86bbedc0ac
function pendulum_rk4(fun, x0, Tf, h)    
    t = Array(range(0,Tf,step=h))
    
    x_hist = zeros(length(x0),length(t))
    x_hist[:,1] .= x0
    
    for k = 1:(length(t)-1)
        x_hist[:,k+1] .= fd_pendulum_rk4(x_hist[:,k], h)
    end
    
    return x_hist, t
end

# ╔═╡ d7ccbcdc-8343-4639-bc3a-92d24c7a6c0c
md"""
 Sim: $(@bind sim2 CheckBox())
"""

# ╔═╡ ebe3f468-e6b7-4afa-8bb6-5ef9ca182e65
begin
	local x0 = [.1; 0]
	x_hist2, t_hist2 = pendulum_rk4(pendulum_dynamics, x0, 100, 0.1)
	if sim2
		plot(t_hist2, x_hist2[1,:], xlabel="t", ylabel="x(t)", label="")
	end
end

# ╔═╡ 9d6ecd2f-e060-45c9-9a99-03f5f530cf2e
begin
	Ad = ForwardDiff.jacobian(x -> fd_pendulum_rk4(x, 0.01), [0; 0])
	norm.(eigvals(Ad))
end

# ╔═╡ ab0a2cc8-a4a4-42a8-a1e4-c104ee2ba995
begin
	local eignorm = zeros(100)
	local h = LinRange(0,1,100)
	for k = 1:length(eignorm)
	    eignorm[k] = max(norm.(eigvals(ForwardDiff.jacobian(x -> fd_pendulum_rk4(x, h[k]), [0; 0])))...)
	end
	plot(h,eignorm, xlabel="Δₜ", ylabel=L"|Λ|_{∞}", label="")
end

# ╔═╡ 81940b23-b05d-4f1b-be82-0c34bd0ad21a
md"""### Implicit Form:

```math
x_{t+1} = f_d(x_{t+1}, u_{t+1})
```

#### Backward Euler Integration:

```math
x_{t+1} = x_t + \underbrace{\Delta_t \cdot f(x_{t+1}, u_{t+1})}_{f_d}
```

"""

# ╔═╡ 1f0b068a-da49-4fc5-a91b-fc6da9ecc434
function pendulum_backward_euler(fun, x0, Tf, dt)
    t = Array(range(0,Tf,step=dt))
    
    x_hist = zeros(length(x0),length(t))
    x_hist[:,1] .= x0
    
    for k = 1:(length(t)-1)
        e = 1
        x_hist[:,k+1] = x_hist[:,k]
        while e > 1e-8
            xn = x_hist[:,k] + dt.*fun(x_hist[:,k+1])
            e = norm(xn - x_hist[:,k+1])
            x_hist[:,k+1] .= xn
        end
    end
    
    return x_hist, t
end

# ╔═╡ 86ce1303-e77c-4b93-a2ed-dc0c54a1f191
md"""
 Sim: $(@bind sim3 CheckBox())
"""

# ╔═╡ b857efd5-dba1-4872-b133-59e80d7cd489
begin
	local x0 = [.1; 0]
	x_hist3, t_hist3 = pendulum_backward_euler(pendulum_dynamics, x0, 10, 0.01)
	if sim3
		plot(t_hist3, x_hist3[1,:], xlabel="t", ylabel="x(t)", label="")
	end
end

# ╔═╡ de4807ca-4e17-4020-9810-5f7c0fcae9a3
question_box(md"### Why most simulators use Backward--Euler?")

# ╔═╡ Cell order:
# ╟─13b12c00-6d6e-11f0-3780-a16e73360478
# ╟─ec473e69-d5ec-4d6a-b868-b89dadb85705
# ╟─8d7a34ef-5a2d-41a8-ac55-39ab00d7e432
# ╟─ced1b968-3ba6-4e58-9bcd-bbc6bee2b93c
# ╟─97994ed8-5606-46ef-bd30-c5343c1d99cf
# ╟─1f774f46-d57d-4668-8204-dc83d50d8c94
# ╟─a0f71960-c97c-40d1-8f78-4b1860d2e0a2
# ╟─1d7092cd-0044-4d38-962a-ce3214c48c24
# ╟─60ba261a-f2eb-4b45-ad6d-b6042926ccab
# ╟─15709f7b-943e-4190-8f40-0cfdb8772183
# ╟─5d7a4408-21ff-41ec-b004-4b0a9f04bb4f
# ╟─7e487ebc-8327-4f3e-a8ca-1e07fb39991a
# ╟─bd623016-24ce-4c10-acb3-b2b80d4facc8
# ╟─2d211386-675a-4223-b4ca-124edd375958
# ╟─45275d44-e268-43cb-8156-feecd916a6da
# ╟─c08f511e-b91d-4d17-a286-96469c31568a
# ╟─b3129bcb-c24a-4faa-a5cf-f69ce518ea87
# ╟─c1f43c8d-0616-4572-bb48-dbb71e40adda
# ╟─57d896ca-221a-4cfc-b37a-be9898fac923
# ╟─52005382-177b-4a11-a914-49a5ffc412a3
# ╟─8ea866a6-de0f-4812-8f59-2aebec709243
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
# ╟─5a691d10-44f7-4d44-a2c9-a7d4d720b7cc
# ╟─f3d155c6-5384-481a-8373-582e753ea8d6
# ╟─ee5c5e2e-e9f1-4f94-95c9-21d506281ae1
# ╟─b9aeab8a-f8ea-4310-8568-5d6bda0bb4d3
# ╟─30a013a8-c02e-4816-af0d-9280473c916b
# ╟─2cc57795-717a-46f0-9bb5-67b601a766de
# ╟─59f6167d-796c-4844-89c0-c796fb59aa2e
# ╟─58c2e1f2-819d-40fc-8e92-03a1a3019a3d
# ╟─70690e72-c31e-4c91-b211-35c74d1d9973
# ╟─5f35a169-887f-477f-b010-167627f7ce4c
# ╟─5c8f6256-e818-4aa1-aea0-02422df8f77c
# ╟─a3f47dad-3cfa-4f6d-9dc6-d4b09d209f86
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
# ╟─876bdea3-9a0e-4e40-9ae4-ef77b08c2428
# ╟─161a2a6e-567f-4994-8d77-9a0f0962cdd9
# ╟─f08c95ef-f784-4c88-be61-91e0549d421b
# ╟─41e1934a-2a43-44c3-9bcf-bef56f4d057e
# ╟─fc585231-a650-4efb-aea4-95110dbf8fa5
# ╟─da8a1e40-7e7c-472a-933a-c585754270bd
# ╟─211f75ed-8acb-4261-bf97-8fc70be2a79b
# ╟─7e7ce801-a510-4ef2-910b-9e10b685ea58
# ╟─7ad76460-e60c-4579-bee0-ac6f3c511877
# ╟─e055fdc3-fb83-4c13-85c9-6edcb661e2dd
# ╟─db1c0c1e-9f3b-44b6-8974-6d5ab5e5e8b7
# ╟─04b6560f-aee2-41fd-86fa-d075f0b3d738
# ╟─c7b11f27-1582-45d0-adef-32eb9c6de588
# ╟─4cd2306d-e3f3-4895-8798-596f6c353bdc
# ╟─ca9d4d0c-40c2-4144-866f-db1417d42c8f
# ╟─3a576353-76bb-4c12-b2a2-b37e8e1dd17f
# ╟─871587c3-380a-4492-b680-aa7b6dd2004f
# ╠═29df2037-456f-4f98-9e32-71037e3d76c4
# ╠═27b490fa-8c2e-4a1a-a6d6-d57ad149a066
# ╠═f6c075f9-9d79-46ba-870e-e12c2b3357df
# ╟─e541f564-a1f1-4fa9-a62c-86265722857a
# ╟─166138f8-4c74-426e-8ce6-2e9c4416de9f
# ╟─facaecf5-2d27-4707-8087-0aa18517d7cd
# ╟─19f3d541-7452-4bcc-89e0-51bf5dab34e6
# ╟─800341e2-de0d-43a4-b0f1-a74021963f43
# ╟─7acd26bc-e35b-47a4-aca3-719f106f3238
# ╟─c94a2f37-9782-4fd8-bae3-61fa8f82ca2d
# ╠═5a1cdae4-2b2f-4df0-866a-6e62be6ddb4a
# ╠═8903c3cc-fc4b-4ffa-bf5b-7645724e8b02
# ╟─bc3b48bd-0a4b-4282-84dd-cbb7bf6b084e
# ╟─7ebc1af0-e8d7-40b2-8395-48aaacb272de
# ╟─cc5a6c0f-bf72-4e5d-aed6-7cbbadac862a
# ╠═f722b8da-0440-4bc1-8124-84305ef4bd4d
# ╠═22ab7266-894c-457c-ad19-1c86bbedc0ac
# ╟─d7ccbcdc-8343-4639-bc3a-92d24c7a6c0c
# ╟─ebe3f468-e6b7-4afa-8bb6-5ef9ca182e65
# ╠═9d6ecd2f-e060-45c9-9a99-03f5f530cf2e
# ╟─ab0a2cc8-a4a4-42a8-a1e4-c104ee2ba995
# ╟─81940b23-b05d-4f1b-be82-0c34bd0ad21a
# ╠═1f0b068a-da49-4fc5-a91b-fc6da9ecc434
# ╟─86ce1303-e77c-4b93-a2ed-dc0c54a1f191
# ╠═b857efd5-dba1-4872-b133-59e80d7cd489
# ╟─de4807ca-4e17-4020-9810-5f7c0fcae9a3
