# Unscented Optimal Control (UOC) Chapter
#
# Nonlinear stochastic optimal control is challenging because standard
# methods like LQG assume linear dynamics or rely on linearization,
# which can introduce significant errors when the system is strongly nonlinear.
# Unscented control (UOC) addresses this by accurately propagating state
# distributions through nonlinear dynamics using sigma points.

# Motivation:
#   - LQG assumes linear system dynamics or uses a first-order approximation.
#   - Robust control ensures stability under worst-case disturbances but
#     may be conservative for nonlinear stochastic systems.
#   - Unscented control provides a middle ground by capturing nonlinear
#     stochastic effects without linearization.

# Problem formulation:
#   Consider a discrete-time nonlinear system with additive process noise:
#       x_{k+1} = f(x_k, u_k) + w_k
#   The goal is to find a control sequence u_0, ..., u_{T-1} that minimizes
#       E[ sum_{k=0}^{T-1} ℓ(x_k, u_k) + ℓ_T(x_T) ]
#   where ℓ is the stage cost and ℓ_T is the terminal cost.
#   The expectation accounts for the stochastic process noise w_k.

# Approaches to nonlinear stochastic optimal control:
# 1. Iterative LQG (iLQG):
#    - Iteratively linearizes the system around a nominal trajectory
#      and solves a sequence of linear-quadratic problems.
#    - Suitable for “mildly nonlinear” problems because it is computationally efficient.
# 2. Sigma-Point Propagation (Unscented Approach):
#    - Uses a deterministic set of points (“sigma points”) that capture
#      the mean and covariance of the system state distribution.
#    - Propagates these points through the nonlinear dynamics to accurately
#      estimate the predicted mean and covariance of the next state.
#    - Provides second-order accuracy in capturing nonlinear stochastic effects.

# Iterative LQG (iLQG):
#   1. Start with an initial nominal trajectory {x_k^0, u_k^0}.
#   2. Linearize the nonlinear dynamics around this trajectory:
#          δx_{k+1} ≈ A_k δx_k + B_k δu_k
#      where A_k = ∂f/∂x and B_k = ∂f/∂u are evaluated at the nominal trajectory.
#      This approximates the nonlinear system locally as linear.
#   3. Quadratically approximate the cost function around the nominal trajectory:
#          J ≈ Σ δx_k' Q_k δx_k / 2 + δu_k' R_k δu_k / 2
#   4. Solve the resulting linear-quadratic problem to compute a feedback law:
#          δu_k = K_k δx_k + k_k
#   5. Update the nominal trajectory with the computed increments:
#          x_k^0 += δx_k,  u_k^0 += δu_k
#   6. Repeat the linearization and optimization until convergence to a
#      locally optimal control trajectory.
#
# Note: The key idea of iLQG is to treat the nonlinear problem as a series
# of linear problems, iteratively refining the trajectory.

# Sigma-Point Propagation:
#   The Unscented Transform allows us to propagate the mean and covariance
#   of a random variable through a nonlinear function without linearizing it.
#
#   Steps:
#   1. Start with state mean x̄ and covariance P_x.
#   2. Generate 2n+1 sigma points for an n-dimensional state:
#          χ_0 = x̄
#          χ_i = x̄ + (sqrt((n+λ) P_x))_i
#          χ_{i+n} = x̄ - (sqrt((n+λ) P_x))_i
#      Here λ is a scaling parameter, and sqrt denotes the matrix square root.
#      These sigma points capture the distribution of the state exactly up
#      to the second moment.
#   3. Propagate each sigma point through the nonlinear dynamics:
#          χ_i' = f(χ_i)
#   4. Recompute predicted mean and covariance:
#          x̄' = Σ W_i^m χ_i'
#          P_x' = Σ W_i^c (χ_i' - x̄')(χ_i' - x̄')'
#      W_i^m and W_i^c are weights for mean and covariance, respectively.
#
#   Key Insight: This method accurately captures nonlinear effects in
#   the evolution of the state distribution and is particularly useful
#   when the system exhibits strong nonlinearities that invalidate
#   linearization assumptions.

# Example application: Nonlinear reactor control
#   Consider a Continuous Stirred Tank Reactor (CSTR) with exothermic
#   reactions and strongly nonlinear kinetics. The control goal
#   is to maintain reactor temperature under process noise and measurement
#   noise.
#
#   - Robust Control (H∞) designed from a fixed linear model may perform
#     poorly as nonlinear effects increase.
#   - Unscented Control uses sigma points to propagate the full nonlinear
#     dynamics and combines this with measurements to estimate the true
#     state.
#   - The control input is then computed based on the sigma-point propagated
#     estimate, leading to improved temperature tracking and reduced
#     sensitivity to measurement noise.

# Summary:
#   Unscented Control provides a principled way to handle nonlinear stochastic
#   systems by maintaining higher-order accuracy in state estimation and
#   control computation. It avoids the pitfalls of linearization and is
#   particularly suited for systems where nonlinearities and stochasticity
#   are significant.

println("Unscented control chapter loaded. This file provides explanations of iLQG and sigma-point propagation methods for nonlinear stochastic control.")
