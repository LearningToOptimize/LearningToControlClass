# Linear Quadratic Gaussian (LQG) Control
#
# LQG control is about designing an optimal controller for linear systems 
# when there is uncertainty in both the system dynamics and the measurements.
# 
# System setup:
#   x_{t+1} = A * x_t + B * u_t + w_t
#   y_t     = C * x_t + v_t
# Here, w_t and v_t are Gaussian process and measurement noise.
#
# The goal is to minimize a quadratic cost function:
#   J = E[ sum(x_t' * Q * x_t + u_t' * R * u_t) ]
# which balances performance (state tracking) and control effort.

using LinearAlgebra

# Linear Quadratic Regulator (LQR)
#
# The LQR computes the optimal control law for a deterministic linear system
# when the full state is known. For continuous-time systems:
#   dx/dt = A*x + B*u
# The quadratic cost is:
#   J = ∫ (x'Qx + u'Ru) dt
# The optimal control law is:
#   u = -K*x
# where K is obtained by solving the continuous-time Riccati equation:
#   A'*P + P*A - P*B*R^-1*B'*P + Q = 0
#
# Here, we implement a discrete-time version.

function dlqr(A, B, Q, R)
    P = copy(Q)
    maxiter = 1000
    tol = 1e-8
    for i in 1:maxiter
        P_new = A' * P * A - A' * P * B * inv(R + B' * P * B) * B' * P * A + Q
        if norm(P_new - P) < tol
            P = P_new
            break
        end
        P = P_new
    end
    K = inv(R + B' * P * B) * B' * P * A
    return K, P
end

# Kalman Filter (State Estimation)
#
# In LQG, the controller does not know the true state x_t.
# Instead, it receives noisy measurements y_t and uses a Kalman filter
# to produce an optimal estimate x_hat_t.
#
# The filter updates the estimate according to:
#   x_hat_{t+1} = A*x_hat_t + B*u_t + L*(y_t - C*x_hat_t)
# where L is the Kalman gain chosen to minimize estimation error variance.

function kalman_filter(A, B, C, Q, R, y)
    n = size(A, 1)
    x_hat = zeros(n, length(y))
    P = zeros(n, n)
    for t in 1:length(y)
        # Prediction step
        if t == 1
            x_pred = zeros(n)
            P_pred = P + Q
        else
            x_pred = A * x_hat[:, t-1]
            P_pred = A * P * A' + Q
        end
        # Update step
        K = P_pred * C' * inv(C * P_pred * C' + R)
        x_hat[:, t] = x_pred + K * (y[t] - C * x_pred)
        P = (I - K * C) * P_pred
    end
    return x_hat
end

# Separation Principle
#
# A key property of LQG control is the separation principle:
# The design of the controller (LQR) and the estimator (Kalman filter)
# can be done independently. Once K and L are computed, the control input is:
#   u_t = -K * x_hat_t
# This allows the system to act as if the true state were known, using only the estimated state.
