# -*- coding: utf-8 -*-
"""
LQG vs H∞ demonstration: H∞ clearly better under persistent disturbance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

np.random.seed(42)

# ---------------------------
# System parameters
# ---------------------------
a = 0.3
b = 1.0
dt = 0.1
T = 50
N = int(T/dt)

# ---------------------------
# Setpoint profile
# ---------------------------
x_set = np.zeros(N)
x_set[int(N*0.1):int(N*0.4)] = 50.0
x_set[int(N*0.4):int(N*0.6)] = 20.0
x_set[int(N*0.6):] = 50.0

# ---------------------------
# Disturbance: structured + Gaussian
# ---------------------------
# Persistent, worst-case disturbance for H∞ to handle
w_structured = 2 * np.sin(0.5*np.arange(N))    # structured
w_noise = 2.0 * np.random.randn(N)               # small Gaussian noise
w_process = w_structured + w_noise

v_meas = 15 * np.random.randn(N)  # measurement noise

# ---------------------------
# LQR gain
# ---------------------------
Q = 1.0
R = 0.1
P = solve_continuous_are(np.array([[-a]]), np.array([[b]]), np.array([[Q]]), np.array([[R]]))
K_lqr = (b * P / R).item()

# ---------------------------
# H∞ gain: scale up to attenuate structured disturbance
# ---------------------------
K_hinf = K_lqr * 2.0  # aggressive gain to fight worst-case disturbance

# ---------------------------
# Simulation
# ---------------------------
x_lqg = np.zeros(N)
x_hat = np.zeros(N)
x_true = 0.0
x_hinf = np.zeros(N)
x_true_hinf = 0.0

for k in range(N-1):
    # --- LQG ---
    y = x_true + v_meas[k]
    # Update estimate
    L = 1  # Kalman-like gain (tuned)
    x_hat[k] = x_hat[k] + dt*(-a*(x_hat[k]-x_set[k]) + b*(-K_lqr*(x_hat[k]-x_set[k])) + L*(y - x_hat[k]))
    # Predict next state
    x_lqg[k+1] = x_hat[k] + dt*(-a*(x_hat[k]-x_set[k]) + b*(-K_lqr*(x_hat[k]-x_set[k])))
    # True system
    x_true = x_true + dt*(-a*(x_true - x_set[k]) + b*(-K_lqr*(x_hat[k]-x_set[k])) + w_process[k])

    # --- H∞ ---
    u_hinf = -K_hinf*(x_true_hinf - x_set[k])
    x_true_hinf = x_true_hinf + dt*(-a*(x_true_hinf - x_set[k]) + b*u_hinf + w_process[k])
    x_hinf[k+1] = x_true_hinf

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(12,6))
plt.plot(x_set, 'k--', label='Setpoint')
plt.plot(x_lqg, label='LQG')
plt.plot(x_hinf, label='H∞ Controller')
plt.xlabel('Time step')
plt.ylabel('Temperature / deviation')
plt.title('LQG vs H∞: Persistent disturbance scenario')
plt.legend()
plt.grid(True)
plt.show()
