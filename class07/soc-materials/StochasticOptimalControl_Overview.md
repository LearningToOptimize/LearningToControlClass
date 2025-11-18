# Class07 Stochastic Optimal Control (SOC)
#### This file provides an introduction to SOC, its motivation, methods, and applications, with formulas explained in context.

#
Stochastic Optimal Control (SOC) is concerned with choosing control actions in systems where both the dynamics and the observations are noisy. 
In real-world systems, uncertainties arise from sensor noise, model inaccuracies, and external disturbances. 
SOC explicitly accounts for these uncertainties while attempting to optimize a performance objective.

Consider a discrete-time system with dynamics given by: $x_{t+1} = A * x_t + B * u_t + w_t$
where $x_t$ represents the state at time $t$, $u_t$ is the control input, and $w_t$ is a stochastic disturbance, typically modeled as a zero-mean Gaussian with covariance $Q_w$. 

Measurements are also noisy:
    $y_t = C * x_t + v_t$
where $v_t$ represents measurement noise with covariance $R_v$. 

The control objective is usually formulated as the minimization of an expected quadratic cost:
    $J = E[ sum_{t=0}^{T-1} (x_t' * Q * x_t + u_t' * R * u_t) ]$
Here, $Q$ and $R$ are weighting matrices that balance the importance of state deviations versus control effort.

This framework allows controllers to explicitly trade off performance and robustness, producing actions that are principled and reliable even under uncertainty.

SOC has broad applications. In aerospace engineering, it is used to stabilize aircraft under turbulence and wind gusts. In robotics, SOC ensures reliable navigation when sensors are noisy or imperfect. In finance, it helps optimize portfolios under stochastic returns. In process engineering, SOC helps control chemical reactors, distillation columns, and energy systems where measurement noise and disturbances are significant.

Traditional deterministic control assumes perfect knowledge of the system state and neglects uncertainty. Controllers designed under this assumption often perform poorly or fail when noise or model mismatch is present. SOC remedies this by explicitly modeling uncertainties and incorporating them into the decision-making process.

Measurement choices directly affect the uncertainties the controller must handle. For example, temperature can be measured using thermocouples, resistance temperature detectors (RTDs), or infrared sensors. Thermocouples are inexpensive and cover a wide temperature range, but they are noisy and less accurate. RTDs are more precise and stable but slower and costlier. Infrared sensors are non-contact and fast, but their readings depend on surface emissivity and line-of-sight. The method chosen determines the type and magnitude of errors the controller will encounter, which influences the design of the control strategy.

Several stochastic control methods exist, each suited to different types of systems and uncertainty. Linear Quadratic Gaussian (LQG) control is optimal for linear systems with Gaussian noise, providing a separation principle between state estimation and control. Robust control, including H-infinity methods, focuses on worst-case scenarios, ensuring stability and bounded performance under model mismatch or bounded disturbances. Unscented Optimal Control (UOC) and iterative Linear Quadratic Gaussian (iLQG) approaches handle nonlinear dynamics and non-Gaussian noise. Sigma-point propagation, used in UOC, accurately tracks the mean and covariance of the state through nonlinear transformations. iLQG iteratively linearizes the system around a nominal trajectory and computes locally optimal control laws.

The choice of method depends on the system's linearity, noise characteristics, uncertainty magnitude, and performance versus safety requirements. There is no single stochastic control method that is universally optimal; the system context and design priorities must guide the selection.

In this chapter, we will examine four major areas. First, LQG control is introduced to illustrate optimal control for linear systems under Gaussian noise and the separation between estimation and control. Second, Kalman filtering is described as a recursive technique for estimating system states from noisy measurements. Third, robust control methods are discussed, contrasting stochastic and worst-case approaches, and introducing H-infinity methods for handling uncertainties and disturbances. Finally, Unscented Optimal Control and iLQG are presented, showing how sigma-point propagation and iterative trajectory optimization allow SOC methods to handle nonlinear stochastic systems effectively.
