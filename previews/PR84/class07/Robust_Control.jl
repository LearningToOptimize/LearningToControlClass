### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 6527df00-c3c9-11f0-13e2-7d05788e7333
md"""
# Robust Control  
### Ensuring stability and performance under **model uncertainty**, **unmodeled dynamics**, and **external disturbances**

In real-world systems including temperature control systems, vehicles, chemical processes, and financial models, our mathematical models are *never* exact.  
Parameters drift, sensors degrade, actuators saturate, and disturbances appear in unexpected ways.

**Robust control** is the branch of control theory that explicitly accounts for these uncertainties and guarantees acceptable performance *even under worst-case conditions*.

This notebook walks through:

1. What model uncertainty means and how it is represented  
2. Stochastic vs worst-case philosophies  
3. LQG and stochastic robustness  
4. H∞ (H-infinity) worst-case robustness  
5. μ-synthesis and structured uncertainty  
6. Robust trajectory optimization  
7. A temperature-control example contrasting LQG and H∞ controllers  
"""


# ╔═╡ efd5146a-fe33-48fb-9048-e60b9373a461
md"""
# ## 1. What Is Model Uncertainty?

Even if we write a model

$x_{k+1} = A x_k + B u_k$

we know that in reality:

- The matrix **A** is not exact  
- The matrix **B** is not exact  
- There are disturbances, offsets, nonlinearities, bias, noise  

This leads to a more realistic model:

$x_{k+1} = (A + \Delta A)x_k + (B + \Delta B)u_k + w_k$

Where:

- **$ΔA, ΔB$**: unknown deviations from the nominal model  
- **$w_k$**: disturbance or process noise  
- **$v_k$**: measurement noise (if sensors are used)  

The goal of robust control is to design a controller **u_k** that works for all allowed $ΔA, ΔB, w_k$.

Different robust-control frameworks differ by **how uncertainty is assumed to behave**.
"""


# ╔═╡ 3fb8940e-3ef7-43b0-90bc-38bd320120e7
md"""
# 2. Two Philosophies of Robustness

Robust control is centered around two fundamentally different interpretations of uncertainty:

---

## **2.1 Stochastic Approaches (example: LQG)**  
These assume that uncertainties behave like **random variables** with known distributions.

Example assumptions:

- Process noise \($w_k \sim \mathcal{N}(0, Q)$\)  
- Measurement noise \($v_k \sim \mathcal{N}(0, R)$\)

You design:

- A **Kalman filter** to optimally estimate the state from noisy measurements  
- An **LQR** controller to compute optimal control inputs  

Together they form the **LQG controller**.

### Strengths
- Optimal when noise statistics are correct  
- Very strong performance in predictable environments  

### Weaknesses
- Not robust to **worst-case disturbances**  
- Sensitive when the real noise distribution deviates from assumptions  

---

## **2.2 Worst-Case Approaches (example: H∞)**  
Here uncertainty is treated as **an adversary** that tries to degrade performance.

Instead of assuming a distribution, we assume a **bound**:


$\|\Delta A\| \le \bar{\Delta A}, \quad \|\Delta B\| \le \bar{\Delta B}, \quad \|w_k\| \le w_{\max}$

The controller is designed so that:

- Even the **worst choice** of disturbances cannot destabilize the system  
- Performance measures remain within acceptable bounds  

### Strengths
- Guaranteed performance under worst conditions  
- Robust to modeling errors  

### Weaknesses
- More conservative  
- Can sacrifice performance during nominal conditions  
"""


# ╔═╡ bb01c629-ffd9-4d40-b505-0a76f779f500
md"""
# 3. Detailed Stochastic Example: LQG Modeling

We begin with a discrete-time stochastic system:

$x_{k+1} = A x_k + B u_k + w_k,
\quad w_k \sim \mathcal{N}(0, Q)$

and noisy observations:

$y_k = C x_k + v_k,
\quad v_k \sim \mathcal{N}(0, R)$

### Inside LQG:

1. **Kalman filter** produces an optimal state estimate  
2. **LQR** computes  
   $u_k = -K \hat{x}_k$
3. LQG = LQR + Kalman filter

This is conceptually robust to *random* noise, but not robust to **bounded, structured uncertainties** in A or B.
"""


# ╔═╡ fd8746c9-42ca-4ecb-8040-17960982b533
md"""
# 4. Worst-Case Formulation: H∞ Control

### Core idea:
The controller competes against an adversarial disturbance that tries to maximize the output energy.

We want the induced L2 gain from disturbance input \($w$\) to regulated output \($z$\) to satisfy:


$\|T_{zw}\|_\infty < \gamma$

Where:

- \($T_{zw}$\) is the closed-loop transfer function  
- \($\gamma$\) is the worst-case amplification bound  

Smaller \($\gamma$\) = more robust to disturbances.

---

## State-space model:

$\dot{x} = A x + B_u u + B_w w$

$z = C_z x + D_{zu} u + D_{zw} w$

$y = C_y x + D_{yu} u + D_{yw} w$

The H∞ control problem finds a controller \($u(t)$\) that makes the system robust against the worst possible disturbance input \($w(t)$\).

This yields:

- Guaranteed stability under model uncertainty  
- Guaranteed bound on performance degradation  
"""


# ╔═╡ 35ae220e-f0e8-407c-b957-5ebf869c0c47
md"""
# ## 5. μ-Synthesis: Handling Structured Uncertainty

### Why H∞ is not enough
H∞ is great for unknown but “norm-bounded” uncertainties (i.e., “anything with size < δ”).  
But many real systems have **structured uncertainty**, e.g.:

- Sensor bias  
- Actuator saturation  
- Temperature-dependent parameters  
- Known block-diagonal uncertainty structures  
- Gain drift only in one channel  

H∞ cannot capture these structures.

---

## μ-synthesis solves:

$\mu < 1$

where **μ** (mu) is the structured singular value.

### What μ measures:
It measures whether structured uncertainty Δ can destabilize the system:

- If μ < 1 → stable under all allowed Δ  
- If μ ≥ 1 → system might become unstable  

---

## D–K Iteration (the μ-synthesis algorithm)

### Step 1: **D-step**
Compute a scaling matrix \($D$\) that upper-bounds the effect of structured uncertainty.

### Step 2: **K-step**
Design an H∞ controller using that scaling.

### Repeat:
D → K → D → K  
until convergence.

μ-synthesis produces controllers that are:

- More robust than H∞  
- Less conservative  
- Explicitly account for structured uncertainty  
"""


# ╔═╡ 82075bd8-4908-4b4e-9f7b-b616386d59a5
md"""
# 6. Robust Trajectory Optimization 

Worst-case robust control can be formulated as a **min-max game**:

$\min_{u_{0:T}} \max_{\Delta A, \Delta B, w_{0:T}} 
\sum_{t=0}^{T} x_t^\top Q x_t + u_t^\top R u_t$

subject to:

$x_{t+1} = (A + \Delta A)x_t + (B + \Delta B)u_t + w_t$

with bounds:

$\|\Delta A\| \le \bar{\Delta A}, \quad
\|\Delta B\| \le \bar{\Delta B}, \quad
\|w_t\| \le w_{\max}$

Interpretation:

- **Controller (minimizer)** chooses u(t) to keep performance good  
- **Nature (maximizer)** chooses worst possible disturbance  

This viewpoint unifies many robust control methods:

- Online robust MPC  
- H∞ design  
- Adversarial learning  
- Distributionally-robust optimization  
"""


# ╔═╡ 3e1b18e2-06b6-4299-8d58-202c99d399ef
md"""
# 7. Temperature Control Example: LQG vs H∞

We consider a scalar thermal process:


$\frac{dx}{dt} = -a(x - x_\text{set}) + b u + w_{\text{process}}$

where:

- \($x$\): system temperature  
- \($x_\text{set}$\): desired temperature  
- \($a > 0$\): thermal dissipation rate  
- \($b$\): actuator effectiveness (heater/cooler)  
- \($w_{\text{process}}$\): external heat disturbances  

### Disturbance model:

$w_{\text{process}}(t) = 2\sin(0.5t) + \text{Gaussian noise}$

### Measurement:
$y = x + v_{\text{meas}}, \quad v_{\text{meas}} \sim \mathcal{N}(0, 15^2)$

---

## **LQG Controller**

- Assumes noise is Gaussian with known covariance  
- Kalman filter estimates the temperature  
- LQR regulates temperature  

**Good performance under expected/stochastic conditions**.

---

## **H∞ Controller**

- Assumes worst-case disturbance with bounded magnitude  
- Gains chosen to attenuate worst disturbances  
- No assumption of sinusoid or Gaussian behavior  

**More conservative, but guaranteed to perform well under extreme disturbance conditions.**  

---

## Outcome Comparison:

### LQG:
- Excellent tracking when noise is reasonably small  
- Vulnerable to persistent sinusoidal disturbances  
- Not robust to model mismatch (e.g., actuator weakening)

### H∞:
- Very stable under persistent oscillations  
- Handles sudden changes and worst-case heat pulses  
- Slightly worse nominal tracking due to conservative nature  
"""


# ╔═╡ bcc3e98a-d94d-4d1c-945c-11f770b12bae
md"""
---
# Final Message
This notebook provides an introduction to robust control, covering:

- Stochastic vs worst-case uncertainty  
- LQG, H∞, μ-synthesis  
- Structured vs unstructured uncertainty  
- Min-max robust trajectory optimization  
- Realistic temperature control example  
"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╠═6527df00-c3c9-11f0-13e2-7d05788e7333
# ╠═efd5146a-fe33-48fb-9048-e60b9373a461
# ╠═3fb8940e-3ef7-43b0-90bc-38bd320120e7
# ╠═bb01c629-ffd9-4d40-b505-0a76f779f500
# ╠═fd8746c9-42ca-4ecb-8040-17960982b533
# ╠═35ae220e-f0e8-407c-b957-5ebf869c0c47
# ╠═82075bd8-4908-4b4e-9f7b-b616386d59a5
# ╠═3e1b18e2-06b6-4299-8d58-202c99d399ef
# ╠═bcc3e98a-d94d-4d1c-945c-11f770b12bae
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
