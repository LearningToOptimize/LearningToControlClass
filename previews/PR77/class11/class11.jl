### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 926b3e8e-31b6-4de0-b3b1-d70294426a1c
begin
	class_dir = @__DIR__
    import Pkg
	Pkg.activate(".")
	Pkg.instantiate()
    using PlutoUI
	using ShortCodes
	import Images: load
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 2000px;
    		padding-left: max(100px, 10%);
    		padding-right: max(100px, 10%);
		}
		.img {
		    text-align: center;
		}
	}

</style>
"""
end

# ╔═╡ c48ccfc8-ab82-11f0-0da1-69057214509e
md"""
# Physics-Informed Neural Networks

This chapter discusses **P**hysics-**I**nformed **N**eural **N**etworks, and in particular the intersection between PINNs and optimal control.
"""

# ╔═╡ 2296bdef-0a38-4e89-a8dc-0fed82e55bde
html"""
<div class="img">
<img src="https://www.mathworks.com/discovery/physics-informed-neural-networks/_jcr_content/mainParsys/columns/e4219b80-580a-4cc2-a14e-84b7087007c5/image.adapt.full.medium.png/1746533098120.png" />
</div>
"""

# ╔═╡ 52383cc7-4628-4653-8f07-217c8b830193
md"""
## Motivation

Why do we need specialized methods for neural networks when physics is involved?

After all, neural networks are **universal approximators**, meaning they can learn any function.

So what happens if we treat a "physics" learning problem like a normal one?
    After all, ignoring decades (centuries?) of prior work has never led anybody astray.
    So let's just take the typical learning approach -- ignore all structure in the problem
        and simply train a gargantuan model on a massive dataset of input-output pairs.
        Given the sheer volume of venture capital dollars betting on this idea,
        it should work, right? Right??
"""

# ╔═╡ ebb07b92-dd4c-4a5d-bb49-3e065c12737f
html"""
<div class="img"><img src="https://www.explainxkcd.com/wiki/images/d/d3/machine_learning.png" /></div>
"""

# ╔═╡ 6c014a4d-909b-459f-aa27-b307c963bef8
md"""
#### A (not so) naive approach

In this section, we'll briefly review the findings of the paper
[What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](https://arxiv.org/abs/2507.06952)


This paper uses state-of-the-art neural network architectures, namely Transformers,
to attempt to learn orbital mechanics from data. Specifically, the authors randomly
sample some initial conditions (masses, positions, relative velocities),
then use Newton's laws to forward-simulate orbits, and collect sequences of positions.


Since the paper is using transformers, the model takes as input a sequence,
and return a sequence. In this case, each sequence refers to a (synthetic) solar system.


The visualization below shows training points (left) and learned trajectories (right).
Interestingly, this paper goes a step further than merely plotting learned trajectories --
in an effort to "distill" what the model has learned, a symbolic regression post-processing
is applied to extract a (relatively) simple force law, which you can see rendered below
the plot. Clearly, it is quite different from what Newton found.
"""

# ╔═╡ 5347fe90-c3d2-48a6-9010-22105cf3bcbc
PlutoUI.LocalResource(joinpath(pwd(), "badforce.mp4"))

# ╔═╡ 5b79beca-f6b1-4fec-87f6-b312b31c3282
md"""

#### What are they doing?

So, even huge, SOTA architectures fail to learn even relatively simple physics.
When something fails spectacularly, I like to take a step back and re-evaluate
what we are trying to do in the first place. So, let's try to sketch the problem at hand.

```math
\begin{align}
\text{find} \quad & \theta \\
\text{such that} \quad & \text{NN}_\theta(x)=\text{Physics}(x) \quad \forall x\in\operatorname{supp}(\mathfrak{D})
\end{align}
```

From UAT, assuming the NN is big enough, we know there exists such as $\theta$.
To solve it, the paper above takes a very straightforward approach, reformulating to:
```math
\begin{align}
\operatorname{min}_\theta \quad & \mathbb{E}_{x\sim\mathfrak{D}} \;\| \text{NN}_\theta(x)-\text{Physics}(x) \|_2^2
\end{align}
```

Furthermore, the paper pre-collects the a finite dataset $\mathcal{D}$,
where each element is $(x,y\coloneqq\text{Physics}(x))$. This lets us ignore that
there is physics involved, and just apply supervised learning:
```math
\begin{align}
\text{min}_\theta \quad & \sum_{(x,y)\in\mathcal{D}} \;\| \text{NN}_\theta(x)-y \|_2^2
\end{align}
```
"""

# ╔═╡ 78e684d9-f914-41e6-82af-fab18bb0aba1
md"""
#### The problem

Clearly, although this approach is simple and straightforward, the results are not good.
In this section, we attempt to explore why that is the case.


An important property of neural networks is their susceptance to
so-called *adversarial examples*. These are points $x$ for which $\text{NN}(x)$ returns a
poor prediction, despite being "in-distribution."

These points arise due to the **high-dimensional** nature of most learning problems,
and the fact that we are using **finite training data**. In high dimension, this means
there are huge regions in $\text{supp}(\mathfrak{D})$ that the model is effectively
"unaware" of. In these regions, good performance can really only be attributed to divine
benevolence (some people like to use empty terms like "implicit regularization" instead,
potató, potáto).
"""

# ╔═╡ 62e09788-c204-4e66-9b2e-19b1c262461c
html"""
<div class="img">
<img style="width: 45%" src="https://gradientscience.org/images/piggie.png" /> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img style="width:48%" src="https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/202108/uncertainty.png?itok=qYxr9l7l" /></div>
"""

# ╔═╡ bd8d65d0-8058-4a2e-8bc3-e7e2d240b8ae
html"""
<sup><sub>I highly recommend visiting <a href="https://en.wikipedia.org/wiki/Flatland">Flatland</a> if you want some intuition for why this is. I like <a href="https://youtu.be/SwGbHsBAcZ0?si=GxzxalPRLmEYjLZA&t=411"> this series of videos </a>.</sub></sup>

<div class="img"><img height=150px src="https://imgs.xkcd.com/comics/flatland.png" /> </div>
"""

# ╔═╡ 1d319400-a7f5-4618-967c-7ba5a7d606dc
md"""
This behavior is only more pronounced in physics-related learning problems. Often,
data comes from expensive and/or cumbersome experiments, meaning training dataset
sizes are in the 100s to 1000s of points, rather than billions to trillions.
Indeed, "serious" applications of PINNs are not trying to reconstruct physical
laws that have been known for centuries -- they are interested in accelerating the
resolution of PDEs, solving inverse problems, etc.

So how can we do better? Can we leverage our expert knowledge to train better models? (Yes!!)
"""

# ╔═╡ bee0d0f9-6e8f-474f-bfa7-cdab6f91619a
md"""
### PDE learning

The most common application of PINNs is to solve parametric partial differential
equations. To analogize back to the orbital mechanics, the PDE is like the ground truth
gravitational physics model, and the parameters correspond to e.g. different masses of planets.
Importantly, *across all parameters/inputs, we know (as experts) that we are looking at the
**same underlying physics**.* Similarly in PDE learning, we are always interested in the **same PDE**
(or at least a family of very similar PDEs).
Let's explore in some more detail this setting, so we can try to figure out what expert
knowledge can be injected into the learning setup.
"""

# ╔═╡ a71a4dc1-1a68-43dd-8cfb-c87178d051a4
md"""
#### Formal setup

We consider a family of problems stated in terms of parameter vector
$\gamma \in \mathcal{P} \subset \mathbb{R}^p$:

```math
\begin{align}
\mathcal{O}\big(u(t,\mathbf{x}); \gamma \big) &= f\big(u(t,\mathbf{x}; \gamma)\big) \quad &&\mathbf{x}\text{ in } \Omega,\; t\in[0,T],\\
\mathcal{O}_\mathcal{B}\big(u(t,\mathbf{x}); \gamma) &= f_B(u(t,\mathbf{x}); \gamma) \quad &&\mathbf{x}\text{ in } \partial\Omega,\; t\in[0,T],\\
\end{align}
```

Here, $t$ is the temporal coordinate, $x$ is the spatial coordinate/state,
and our neural network will learn to approximate the ground-truth solution $u(t,\mathbf{x})$.


Our goal is to fit a neural network $\hat{u}(t,\mathbf{x})$ that respects these conditions.
In other words, we are solving: 

```math
\begin{align}
\text{find} &\enspace \theta \\
\text{such that} &\enspace
\mathcal{O}\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big) = f\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big)  \quad \forall \mathbf{x} \in\Omega,\; t\in[0,T],\\
&\enspace \mathcal{O}_\mathcal{B}\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big) = f_B\big(\hat{u}_\theta(t,\mathbf{x}); \gamma\big)  \quad \forall \mathbf{x}\in\partial\Omega,\; t\in[0,T].
\end{align}
```

Note that this is much more detailed than the orbital mechanics setup used in the paper above.
There, they pretend they don't know physics and try to recover it --
here, we know the physics and try to accelerate solving it.
"""

# ╔═╡ 5b4dc006-0d4f-451f-b16d-c2f755a58b2f
md"""
While useful to write down, this notation isn't exactly helpful for building intuition.
So let's consider a simple example, the heat equation, to get a better sense of
    what is going on:

```math
\begin{align}
\frac{\partial u}{\partial t} = \lambda \frac{\partial^2 u}{\partial \mathbf{x}^2} \quad\quad  &\forall \mathbf{x}\in\Omega,\;\forall t\\
u(t,\mathbf{x})=\rho  \quad\quad&\forall \mathbf{x}\in\partial\Omega,\;\forall t \\
u(0,\mathbf{x})=\gamma  \quad\quad&\forall \mathbf{x}\in\Omega
\end{align}
```
Here, $t$ is time, $\mathbf{x}$ is some coordinates, $\Omega$ is some surface that
we are heating, that has heat parameter $\lambda$, and $\partial\Omega$ is the boundary
of this surface. We further specify the initial condition that the entire surface
has heat $\gamma$ at time zero, and that the boundaries of the surface always
have temperature $\rho$.

Our goal is to fit a neural network $\hat{u}(t,\mathbf{x})$ that
respects these conditions, so:
```math
\begin{align}
\text{find} &\enspace \theta \\
\text{such that} &\enspace
\frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} = \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2}  &\forall \mathbf{x} \in\Omega, \;\forall t\\
&\enspace \hat{u}_\theta(t,\mathbf{x})=\rho  &\forall \mathbf{x}\in\partial\Omega, \;\forall t \\
&\enspace \hat{u}_\theta(0,\mathbf{x})=\gamma  &\forall \mathbf{x}\in\Omega \\
\end{align}
```

Now, it should be evident how the PDE learning problem is more complicated
than the formulation considered in the orbital mechanics learning paper.
Indeed, the first constraint (and in general, the BC/IC also) involves
derivatives of the neural network $\hat{u}$. Although it certainly looks scary,
it turns out the same straightforward approach can be used; just move
all the constraints to the loss. In other words, solve instead

```math
\begin{align}
\text{min}_\theta \quad
&\mathbb{E}_{\mathbf{x}\in\Omega,\,t\in\mathcal{T}} \;\| \frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} - \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2} \|_2^2\\
&\quad+\mathbb{E}_{\mathbf{x}\in\partial\Omega,\,t\in\mathcal{T}} \;\| \hat{u}_\theta(t, \mathbf{x}) - \rho \|_2^2\\
&\quad\quad+\mathbb{E}_{\mathbf{x}\in\Omega} \;\| \hat{u}_\theta(0, \mathbf{x})-\gamma \|_2^2 \\
\end{align}
```

"""

# ╔═╡ 50d2d118-1aae-4a7a-a462-d27270cb2fd0
md"""
Note that we have three different expectations -- the first is over "collocation points",
the second over "boundary condition points" and the third over "initial condition" points.
The figure below visualizes these:
"""

# ╔═╡ 8320a531-d420-449d-89e4-8394e2cf990f
html"""
<div class="img"><img width=600px src="https://ars.els-cdn.com/content/image/1-s2.0-S0378779622005855-gr3_lrg.jpg" /></div>
"""

# ╔═╡ 0cfd0f95-ca95-4b68-a6d4-b09f3c7ed6c4
md"""
**This is the key feature of PINNs: exploiting the availability of collocation points.**
After all, in most learning settings, i.e. image classification, we can't say much about
some random data point. But in the PINN setting, we know that physics holds **everywhere**
(within $\Omega$), even if we don't know the $u$ for that point. Thus, we effectively have
unlimited semi-labeled data, where **the physics itself is the label.**
"""

# ╔═╡ 41b2e013-a9bc-47cf-81ae-d34f93b4b5aa
md"""
It is important to notice that there are no "labels" here so far. In some cases,
measurements $u^\star$ are added:

```math
\begin{align}
\text{min}_\theta \quad
&\sum_{(t,\mathbf{x},u^\star)\in\mathcal{D}}\;\|\hat{u}_\theta(t,\mathbf{x}) - u^\star\|\\
&\quad+\mathbb{E}_{\mathbf{x}\in\Omega,\,t\in\mathcal{T}} \;\| \frac{\partial \hat{u}_\theta(t,\mathbf{x})}{\partial t} - \lambda \frac{\partial^2 \hat{u}_\theta(t,\mathbf{x})}{\partial \mathbf{x}^2} \|_2^2 \\
&\quad\quad+\mathbb{E}_{\mathbf{x}\in\partial\Omega,\,t\in\mathcal{T}} \;\| \hat{u}_\theta(t, \mathbf{x}) - \rho \|_2^2 \\
&\quad\quad\quad+\mathbb{E}_{\mathbf{x}\in\Omega} \;\| \hat{u}_\theta(0, \mathbf{x})-\gamma \|_2^2 \\
\end{align}
```

This is particularly common in settings where the physics is not (entirely)
known -- for example, if we had to estimate $\gamma$ or $\rho$ from data. 
"""

# ╔═╡ 46d0cbc9-d397-4fa5-98ab-450047b44d2c
md"""
## Isn't this a control class?

While using neural networks to solve parametric differential equations is interesting, 
it is only half the battle in optimal control. In the setting of optimal control,
we are interested in *PDE-constrained optimization*, i.e.

```math
\begin{align*}
\min \quad & \Psi(u;\gamma) \\
\text{s.t.} \quad & \mathcal{O}\big(u(t,\mathbf{x}); \gamma \big) = f\big(u(t,\mathbf{x}; \gamma)\big) \quad &&\mathbf{x}\text{ in } \Omega,\; t\in[0,T],\\
&\mathcal{O}_\mathcal{B}\big(u(t,\mathbf{x}); \gamma) = f_B(u(t,\mathbf{x}); \gamma) \quad &&\mathbf{x}\text{ in } \partial\Omega,\; t\in[0,T],\\
\end{align*}
```
"""

# ╔═╡ f5c5224e-05db-424f-b05a-7e8101acadc1
md"""
Now, we are not only interested in approximating the solution to a PDE.
We are now trying to *select a PDE from a family of PDEs* where to evaluate each PDE,
we need the solution. The rest of the chapter reviews a few interesting ways that
select prior works have addressed this problem.
"""

# ╔═╡ b504d8d1-63d6-41e0-bd1e-d946c43d7b9e
md"""

## ICRNN: Optimal Control Via Neural Networks: A Convex Approach


First introduced in [Input Convex Neural Networks](https://arxiv.org/pdf/1609.07152),
ICNNs are neural networks which are *not* universal approximators. Instead, they,
by design, can only approximate convex functions. They achieve this by maintaining two
properties of the architecture: non-negative weights, and convex increasing activation
functions. Under these assumptions, a neural network layer is just a nested series of
positive combinations of convex functions, which is known to be convex. Then, if one
knows that the underlying function to be learned is convex, ICNNs are a powerful tool
    for reducing the hypothesis space of the neural network training.

In the control setting, ICNNs have been leveraged by the paper 
[Optimal Control Via Neural Networks: A Convex Approach](https://arxiv.org/abs/1805.11835),
which proposes an RNN (recursive neural network) version of ICNNs to address control problems.
The figure below shows the regular ICNN (a) and the recursive version (b) proposed by the paper.
"""

# ╔═╡ 9df77ef3-167f-45c6-b41c-26537646dc94
(load(joinpath(class_dir, "icrnn.png")))

# ╔═╡ 10d49868-5668-4c95-93a8-0a94180c6cba
md"""
Note that ICNNs often include skip connections (whose weights are denoted $D_i$ in the figure).
This idea is used to implement the recursive version as well. In particular, the latent state
of each time step is skip-connected into the next time step, allowing the model to utilize
previous activations to inform next predictions -- all while maintaining convexity of the
learned function. Note that only the $W$ weights must be non-negative. Indeed, note that
the first layer weights are effectively $[I, -I]$ since the input is $[u, -u]$
instead of just $u$.
"""

# ╔═╡ 9ac946a3-8475-4edd-8226-e2c17dae42c8
md"""
The way the ICRNN paper then uses the neural network is quite interesting.
First, they train using end-to-end data. For example, to control a building's HVAC
system -- which is of course govenered by some underlying (very complicated) PDE,
one starts by collecting measurements of states, controls, and costs. Then, an
ICRNN is trained to fit the overall cost function. Basically, the paper proposes
to replace with learning the entire control problem except for the `min` part.
The minimization is still performed explicitly, as decsribed next.

The ICRNN is then used inside of a Model Predictive Control (MPC) loop,
where the usual local model optimization step (i.e., the form+solve QP step)
is replaced by instead **optimization over** the trained neural network.
Note that this is a different way of using neural networks than most works.
Instead of merely using it as an input-output black box, the paper leverages
the fact that the learned function is convex to enable efficient
(gradient-descent-based) optimization over it.

The results are shown below. As you can see, the approach is very performant,
basically matching the ground-truth.
"""

# ╔═╡ 94d37a10-f30c-4b48-a30d-b593f6edcace
(load(joinpath(class_dir, "icrnn_results.png")))

# ╔═╡ 74d93555-e4e5-4aaf-ac5a-c1224f98d93a
md"""
## PINC -- Physics Informed Neural Networks for Control

Although the ICRNN approach is creative and works for what it was designed for,
it operates in the setting of unknown dynamics. What if you know your dynamics
(approxiamtely)? The paper
[Physics-Informed Neural Nets for Control of Dynamical Systems](https://arxiv.org/abs/2104.02556)
addresses this setting.

In particular, PINC aims to accelerate MPC-based control systems.
The main idea is that, unlike the basic PINN-for-control setup where one learned
to approximate the dynamics then incorporates the neural network into a PDE solver,
PINC proposes to immediately learn the integrator itself. Thus, instead of just
taking as input the state/control, PINC also takes the number of time steps to
look ahead $t$ as input. This enables running an entire MPC inner-loop in
one neural network inference, since one can pass $t=T$. Refer to the
figure below for a reminder of how MPC works.
"""

# ╔═╡ 16049028-7ec9-4932-b52c-56b015eb979f
(load(joinpath(class_dir, "pincmpc.png")))

# ╔═╡ 68bc0e5c-dd7b-4752-a67f-917f97b16067
md"""
The main idea of PINC is train a neural network that can jump ahead,
directly from the $y(0), u[k]$ input to the $y[k+1]$ output. By leveraging the fact
that we know exactly the dynamics in this setting (we may not know the true dynamics,
but within MPC, we operate as if the local model is exact), we can train the model
using a PINN loss, entirely self-supervised. Then, the network can be applied in a
"self-loop" to compute the full horizon up to $y[M]$.


The figure below visualizes the architecture.
"""

# ╔═╡ e21b02a8-90dc-48cf-a1c8-0f81599fbb57
(load(joinpath(class_dir, "pincloop.png")))

# ╔═╡ 08614d3e-244f-46d7-ac38-18c8f1651924
md"""
It is important to highlight that during inference, $T$ is always plugged in for
the $t$ input, while during training, all values of $t\in[0,T]$ are used. This is
necessary since the physics we know is instantaneous -- we can't say anything about
long-term evolutions based on the dynamics alone (i.e. without an integrator). Thus,
the loss is exactly the physics informed loss from above: one term for penalizing the
initial condition, with an expectation over all space, another term for penalizing the
dynamics, with an expectation for all space and time, and another term for the boundary
condition, with an expectation over boundary space and all time.
"""

# ╔═╡ f238f923-3678-456e-bbd0-4936facc65fb
md"""
## Control PINNs: Physics-informed neural networks for PDE-constrained optimization and control

The paper [Physics-informed neural networks for PDE-constrained optimization and control](https://arxiv.org/abs/2205.03377) 
introduces Control PINNs, which take the PINC approach even one step further.
In PINC, as described above, we still have an MPC loop, we just
replace both the integrator and the dynamics with a neural network.
In Control PINNs, the authors propose to replace the entire PDE-constrained optimization
with a neural network. Thus, it is more similar to the ICRNN paper, where we replace all
but the optimization part of the PDE-constrained control problem. However, in Control PINNs,
instead of learning a function to optimize over, the authors propose to learn a
neural network that learns the optimization component as well.

This complicates the learning since now, we must not only incorporate the physics
of the PDE, but also the ``optimization physics,'' i.e. the KKT conditions.
In particular, note that any optimal solution to the PDE-constrained optimization
problem satisfies the following equaitons; now in terms of the state, control,
and *adjoint variables* $\lambda$ (in optimization terms, dual variables):

```math
\begin{align}
    \begin{split}
        &\frac{\partial \mathbf{y}}{\partial t} = 
        \mathbf{f}\bigl(\mathbf{y}, \mathbf{u}\bigr),
        \quad \forall\, t \in [t_0,t_f],\;
        \forall\, x \in \Omega, \\
        &\mathbf{y}\big|_{t_0} = \mathbf{y}_0,
        \quad \forall\, x \in \Omega,\\
        &B\,\mathbf{y} = \mathbf{b}(t, x),
        \quad \forall\, t \in [t_0,t_f],\;
        \forall\, x \in \partial\Omega,
    \end{split}\\[1ex]
    \begin{split}
        &\frac{\partial \boldsymbol{\lambda}}{\partial t} =
        - \boldsymbol{\lambda}^T 
        \frac{\partial \mathbf{f}}{\partial \mathbf{y}}
        \bigl(\mathbf{y}, \mathbf{u}\bigr)
        - \frac{\partial g}{\partial \mathbf{y}}
        \bigl(\mathbf{y}, \mathbf{u}\bigr),
        \quad \forall\, t \in [t_0,t_f],\;
        \forall\, x \in \Omega,\\
        &\boldsymbol{\lambda}\big|_{t_f} =
        w_{\mathbf{y}}\bigl(\mathbf{y}\big|_{t_f}\bigr),
        \quad \forall\, x \in \Omega, \\
        &B^{*}\boldsymbol{\lambda} = 0,
        \quad \forall\, t \in [t_0,t_f],\;
        \forall\, x \in \Omega,
    \end{split}\\[1ex]
    \boldsymbol{\lambda}^T
    \frac{\partial \mathbf{f}}{\partial \mathbf{u}}
    \bigl(\mathbf{y}, \mathbf{u}\bigr)
    + \frac{\partial g}{\partial \mathbf{u}}
    \bigl(\mathbf{y}, \mathbf{u}\bigr)
    = \mathbf{0},
    \quad \forall\, t \in [t_0,t_f],\;
    \forall\, x \in \Omega.
\end{align}
```
"""

# ╔═╡ b661a654-e35b-421e-a42a-c93084146de6
md"""
The rest of the methodology is fairly straight-forward. Indeed, the main concept of PINNs
is to add penalty terms for conditions that we know hold over the entire domain.
In the original description, we include in this list of conditions only the PDE itself.
In Control PINNs, we augment this list of conditions to also include the KKT conditions
listed above. In this way, the method is rather straightforward. In particular,
it avoids the system complexity of having optimization and learning interact in the
control system, instead moving all the complexity into the learning. Despite the
simplicity in design, as we will see next, this leads to very unstable training.
"""

# ╔═╡ f8b5b858-b133-4459-934f-59b7a4b36542
md"""
The figure below shows training curves for a very simple 1D heat example.
Despite the very simple example (which can be addressed by traditional methods easily),
Control PINNs are already struggling to converge nicely. As you can see, even after 2000 epochs,
the Control PINN PDE loss (the error of approximating the PDE dynamics) remains extremely high.
In particular, the loss *increases* extremely quickly between epochs 50 and 400, showing
that the training is having trouble balancing the multitude of loss terms. In fact,
it is clear form the figure that this increase in PDE loss is compensated by the decrease
in boundary and initial condition losses. Although it is important to satisfy boundary
and initial conditions, trading accurate dynamics learning for that is not acceptable.
Thus, although the overall training seems to have converged when looking at the total
loss, the actual downstream performance of the control chosen by the Control PINN cannot
be expected to perform well. The paper itself shows that the controls learned by the
model are very different from the optimal ones.
"""

# ╔═╡ 8635dde1-da60-4f36-aee1-68525e067a9f
(load(joinpath(class_dir, "controlpinnloss.png")))

# ╔═╡ f06720a3-76cc-4cc7-b9aa-59098ec6eefa
md"""
## Making PINNs Work: Balancing Loss Terms


A key weakness of PINNs is that the main idea of the approach -- augmenting the
loss to penalize properties we know to be true -- leads to unstable training.
Many works in the field have shown that in the context of PDEs and control, the
neural network training often ends up prioritizing some of the terms over the others,
due to differences in gradient magnitude and direction. Thus, substantial research effort
has been devoted to figuring out how to better train networks with multiple losses, beyond
just adding the terms together. In
[When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective](https://arxiv.org/pdf/2007.14527),
the authors propse an NTK-based method for computing weights for the loss terms, to more evenly
consider their contributions when training. However, as noted in the paper, even this relatively
complicated approach does not solve all the problems since at the end of the day, it is merely
re-weighting terms, and thus cannot resolve the spectral bias issue.

In [ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/abs/2408.11104),
the authors propose the ConFIG method for combining multiple loss terms. Instead of computing weights,
in ConFIG, the gradient of each of the loss terms is adjusted such that the dot product between
each loss term and the final gradient update is positive, ensuring that each step makes progress
on each loss term. However, the approach is computationally expensive when there are more than
two loss terms. The figure below provides some visual intuition for the ConFIG approach.
"""

# ╔═╡ 97d60aec-2a3d-4206-8a9e-49b14bad7f6d
(load(joinpath(class_dir, "config.png")))

# ╔═╡ 1c88dfa6-5950-478a-a9a0-4d8c515f12f5
md"""
The paper [Modelling of Underwater Vehicles using Physics-Informed Neural Networks with Control](https://arxiv.org/abs/2504.20019) 
provides a nice case-study of using Control PINNs together with several of these
complicated approaches for making PINNs work on real-world problems, in this case for
controlling underwater robots. In this setting, the authors found that a naïve normalization
scheme outperforms ConFIG. In fact, using ConFIG leads to worse long-rollout performance than
not using it; during training, the model is trained for time steps from one to five seconds.
Under the ConFIG loss, when rolling out the model's prediction, it becomes unreliable after
just 0.5-1 seconds, compared to the naïve normalization which remains reliable for 1-1.5 seconds.


[Experiences with Physics-Informed Neural Networks for Optimal Control Problems](https://www.bibliothek.tu-chemnitz.de/ojs/index.php/GAMMAS/article/view/813)
further explores using PINNs for Control, and also finds that PINNs are very difficult
to get working, even when devoting substantial manual effort to tuning the
configurations of the training. The main observation in that work is that the only reliable
technique is **reducing the number of loss terms by designing the neural network
architecture such that it guarantees properties by construction.** For example, if one of
the conditions is that the sum of the neural network outputs should equal one, adding a
softmax layer to the output vastly outperforms adding a $\|\sum(\hat{y})-1\|$ term
to the loss function.



[Physics-Informed Neural Networks with Hard Nonlinear Equality and Inequality Constraints](https://arxiv.org/abs/2507.08124)
proposes a generic approach which embodies this observation.
They propose KKT-HardNet, which incorporates the KKT conditions directly into
the neural network architecture by using a "Newton Layer" that calls Newton's
method for solving nonlinear equations as part of the forward pass. This ensures
that key properties are directly "baked in" to the neural network architecture,
leaving only the data loss for learning. This results in a much more stable and fast
training in terms of number of epochs to obtain good performance. However, its
reliance on the relatively expensive (compared to just matrix multiplaction) Newton
method means that each forward pass takes much more time. Furthermore, the implementation
relies on automatic differentiation through Newton iterates which, as discussed in Class 10,
is often not stable (the implicit function theorem should be used instead).
"""

# ╔═╡ Cell order:
# ╟─926b3e8e-31b6-4de0-b3b1-d70294426a1c
# ╟─c48ccfc8-ab82-11f0-0da1-69057214509e
# ╟─2296bdef-0a38-4e89-a8dc-0fed82e55bde
# ╟─52383cc7-4628-4653-8f07-217c8b830193
# ╟─ebb07b92-dd4c-4a5d-bb49-3e065c12737f
# ╟─6c014a4d-909b-459f-aa27-b307c963bef8
# ╟─5347fe90-c3d2-48a6-9010-22105cf3bcbc
# ╟─5b79beca-f6b1-4fec-87f6-b312b31c3282
# ╟─78e684d9-f914-41e6-82af-fab18bb0aba1
# ╟─62e09788-c204-4e66-9b2e-19b1c262461c
# ╟─bd8d65d0-8058-4a2e-8bc3-e7e2d240b8ae
# ╟─1d319400-a7f5-4618-967c-7ba5a7d606dc
# ╟─bee0d0f9-6e8f-474f-bfa7-cdab6f91619a
# ╟─a71a4dc1-1a68-43dd-8cfb-c87178d051a4
# ╟─5b4dc006-0d4f-451f-b16d-c2f755a58b2f
# ╟─50d2d118-1aae-4a7a-a462-d27270cb2fd0
# ╠═8320a531-d420-449d-89e4-8394e2cf990f
# ╟─0cfd0f95-ca95-4b68-a6d4-b09f3c7ed6c4
# ╟─41b2e013-a9bc-47cf-81ae-d34f93b4b5aa
# ╟─46d0cbc9-d397-4fa5-98ab-450047b44d2c
# ╟─f5c5224e-05db-424f-b05a-7e8101acadc1
# ╟─b504d8d1-63d6-41e0-bd1e-d946c43d7b9e
# ╟─9df77ef3-167f-45c6-b41c-26537646dc94
# ╟─10d49868-5668-4c95-93a8-0a94180c6cba
# ╟─9ac946a3-8475-4edd-8226-e2c17dae42c8
# ╟─94d37a10-f30c-4b48-a30d-b593f6edcace
# ╠═74d93555-e4e5-4aaf-ac5a-c1224f98d93a
# ╟─16049028-7ec9-4932-b52c-56b015eb979f
# ╟─68bc0e5c-dd7b-4752-a67f-917f97b16067
# ╟─e21b02a8-90dc-48cf-a1c8-0f81599fbb57
# ╟─08614d3e-244f-46d7-ac38-18c8f1651924
# ╟─f238f923-3678-456e-bbd0-4936facc65fb
# ╟─b661a654-e35b-421e-a42a-c93084146de6
# ╟─f8b5b858-b133-4459-934f-59b7a4b36542
# ╟─8635dde1-da60-4f36-aee1-68525e067a9f
# ╟─f06720a3-76cc-4cc7-b9aa-59098ec6eefa
# ╟─97d60aec-2a3d-4206-8a9e-49b14bad7f6d
# ╟─1c88dfa6-5950-478a-a9a0-4d8c515f12f5
