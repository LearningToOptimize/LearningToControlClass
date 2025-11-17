## Kalman Filtering
# Goal: Estimate the true state of a system when measurements are noisy.
#
# In many practical systems, we cannot directly observe the full state x_t.
# Instead, we have noisy measurements y_t. The Kalman filter provides the
# optimal linear estimate of the hidden state, assuming linear dynamics
# and Gaussian noise.

using LinearAlgebra

# The Kalman filter operates in two steps: prediction and update.

## Prediction Step:
# Forecast the next state based on the previous estimate and the system model.
# x_pred = A * x_prev + B * u_prev
#   - This uses the known dynamics of the system (A, B) and the previous control input u_prev
# P_pred = A * P_prev * A' + Q
#   - The uncertainty (covariance) is propagated forward
#   - Q represents process noise: uncertainty in how the system evolves
#   - Larger P_pred means we are less confident in the prediction

## Update Step:
# Correct the prediction using the latest measurement y
# K = P_pred * C' * inv(C * P_pred * C' + R)
#   - The Kalman gain K determines how much we trust the measurement vs. prediction
# x_hat = x_pred + K * (y - C * x_pred)
#   - Innovation (y - C*x_pred) adjusts our estimate based on new information
# P = (I - K * C) * P_pred
#   - Covariance decreases after incorporating measurement, increasing confidence

function kalman_filter(A, B, C, Q, R, y, u=nothing)
    n = size(A,1)
    N = length(y)
    x_hat = zeros(n, N)      # estimated states
    P = zeros(n,n)           # covariance
    I = Matrix(I, n, n)

    # Handle optional control input
    u = isnothing(u) ? zeros(size(B,2), N) : u

    for k in 1:N
        # Prediction
        if k == 1
            x_pred = zeros(n)  # initial guess for first step
            P_pred = P + Q
        else
            x_pred = A * x_hat[:,k-1] + B * u[:,k-1]
            P_pred = A * P * A' + Q
        end

        # Update
        K = P_pred * C' * inv(C * P_pred * C' + R)
        x_hat[:,k] = x_pred + K * (y[k] - C * x_pred)
        P = (I - K * C) * P_pred
    end

    return x_hat
end

## Example usage:
# A simple demonstration with a small system. This shows how the filter
# reconstructs the true state from noisy measurements.

if abspath(PROGRAM_FILE) == @__FILE__
    # Define system matrices (example: 2-state thermal system)
    A = [1.0 1.0; 0.0 1.0]   # state transition
    B = [0.0; 1.0]           # control input
    C = [1.0 0.0]            # measurement matrix
    Q = 0.01 * I(2)          # process noise covariance
    R = 0.1                  # measurement noise covariance

    # Simulated noisy measurements
    true_states = [20.0, 21.0, 23.0, 22.0]
    y = true_states .+ randn(length(true_states)) * sqrt(R)

    # Run Kalman filter
    x_hat = kalman_filter(A, B, C, Q, R, y)

    println("Noisy measurements: ", y)
    println("Estimated states: ", x_hat)
end
