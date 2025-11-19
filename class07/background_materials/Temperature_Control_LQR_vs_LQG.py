import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # reproducible

# -------------------------------
# System Definition (1D)
# -------------------------------
A = 0.9       # state transition
B = 0.5       # control input effect

# LQR cost weights (tune Q,R to change aggressiveness)
Q = 1.0
R = 0.1

# Simulation parameters
N = 60
x0 = 60.0  # initial temperature in °C

# Noise parameters
process_noise_std = 0    # small non-zero process noise
measurement_noise_std = 5.0

# Step changes in setpoint
setpoint = np.zeros(N)
setpoint[:20] = 60.0    # start at 60°C
setpoint[20:40] = 80.0  # step to 80°C
setpoint[40:] = 70.0     # step to 70°C

# -------------------------------
# Compute LQR gain (discrete-time)
# -------------------------------
P = Q
for _ in range(200):
    P = Q + A**2 * P - (A * P * B)**2 / (R + B**2 * P)
K = (B * P * A) / (R + B**2 * P)
print("LQR gain K =", K)

# Actuator limits (example)
u_min, u_max = -50.0, 50.0

# -------------------------------
# Naive LQR (reacts directly to noisy measurements)
# -------------------------------
x_lqr = np.zeros(N)
u_lqr = np.zeros(N)
x_true_lqr = x0
y_meas_lqr = np.zeros(N)

for k in range(N):
    # Measurement
    y = x_true_lqr + np.random.randn() * measurement_noise_std
    y_meas_lqr[k] = y
    
    # Control reacts directly to noisy measurement error (no estimator)
    u = -K * (y - setpoint[k])
    # saturate actuator
    u = np.clip(u, u_min, u_max)
    u_lqr[k] = u
    
    # State evolves (process noise)
    x_true_lqr = A * x_true_lqr + B * u + np.random.randn() * process_noise_std
    x_lqr[k] = x_true_lqr

# -------------------------------
# LQG (with Kalman filter) -- FIXED: include control in prediction
# -------------------------------
x_lqg = np.zeros(N)
x_hat = x0        # initial estimate = initial temperature
P_est = 1.0
u_lqg = np.zeros(N)
x_true = x0
y_meas_lqg = np.zeros(N)
u_prev = 0.0

for k in range(N):
    # Measurement
    y = x_true + np.random.randn() * measurement_noise_std
    y_meas_lqg[k] = y
    
    # Prediction (include previous control!)
    x_pred = A * x_hat + B * u_prev
    P_pred = A**2 * P_est + process_noise_std**2
    
    # Kalman update
    K_kalman = P_pred / (P_pred + measurement_noise_std**2)
    x_hat = x_pred + K_kalman * (y - x_pred)
    P_est = (1 - K_kalman) * P_pred
    
    # LQR control based on estimated state
    u = -K * (x_hat - setpoint[k])
    u = np.clip(u, u_min, u_max)
    u_lqg[k] = u
    
    # Apply control to true system (with process noise)
    x_true = A * x_true + B * u + np.random.randn() * process_noise_std
    x_lqg[k] = x_true
    
    # Save u for next prediction
    u_prev = u

# -------------------------------
# Plot results
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(x_lqr, label='Naive LQR (reacts to noisy measurements)', color='red')
plt.plot(x_lqg, label='LQG (state estimated via Kalman filter)', color='blue')
plt.plot(setpoint, 'k--', label='Setpoint', alpha=0.8)
plt.plot(y_meas_lqg, 'kx', alpha=0.4, label='Measurements')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('LQR vs LQG: Tracking Step Changes in Temperature (fixed Kalman prediction)')
plt.legend()
plt.grid(True)
plt.show()
