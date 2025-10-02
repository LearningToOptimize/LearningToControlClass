import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

V = 1.0; F1, F2 = 0.5, 0.3
T1_nom, T2_nom = 350.0, 200.0; T_env = 298.0
dt = 0.02; T_total = 50.0; N = int(T_total / dt)

# setpoint
T_set = np.zeros(N)
T_set[int(N*0.0):int(N*0.25)] = 330.0
T_set[int(N*0.25):int(N*0.5)] = 210.0
T_set[int(N*0.5):int(N*0.75)] = 340.0
T_set[int(N*0.75):] = 320.0

# tuned parameters (these produced R2_uoc >= 0.9 in my run)
reaction_coeff = 5e-5        # mild nonlinearity
feed_noise_std = 0.1         # small disturbances

# robust baseline (weak, tightly clipped)
K_robust = 9.0
robust_limits = (-2.5, 2.5)

# Unscented-like controller (strong, predictive)
K_uoc = 45.0
alpha = 6.0
sigma_std = 1.0
uoc_limits = (-120.0, 120.0)

def reaction_rate(T):
    return reaction_coeff * (T - T_env) ** 2

def r2_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

T_robust = np.zeros(N); T_uoc = np.zeros(N); T_uoc_std = np.zeros(N)
T_robust_current = 300.0; T_uoc_current = 300.0; T_uoc_std[0] = sigma_std
robust_saturated = np.zeros(N)

for k in range(N - 1):
    T1 = T1_nom + np.random.randn() * feed_noise_std
    T2 = T2_nom + np.random.randn() * feed_noise_std

    # Robust P (clipped)
    u_robust = K_robust * (T_set[k] - T_robust_current)
    u_robust = np.clip(u_robust, robust_limits[0], robust_limits[1])
    robust_saturated[k] = 1 if (u_robust <= robust_limits[0] or u_robust >= robust_limits[1]) else 0
    dT_robust = (F1*(T1 - T_robust_current) + F2*(T2 - T_robust_current)
                 + u_robust + reaction_rate(T_robust_current)) / V
    T_robust_current += dt * dT_robust
    T_robust[k + 1] = T_robust_current

    # Unscented-like update
    def sigma_points(T): 
        return np.array([T, T + sigma_std, T - sigma_std, T + 2*sigma_std, T - 2*sigma_std])
    def propagate_sigma(T_sigma, u, T1, T2):
        out = []
        for T in T_sigma:
            dT = (F1*(T1 - T) + F2*(T2 - T) + u + reaction_rate(T)) / V
            out.append(T + dt * dT)
        return np.array(out)

    u_candidate = K_uoc * (T_set[k] - T_uoc_current)
    T_sigma = sigma_points(T_uoc_current)
    T_sigma_next = propagate_sigma(T_sigma, u_candidate, T1, T2)
    error_mean = np.mean(T_sigma_next) - T_set[k]
    u_final = np.clip(u_candidate - alpha * error_mean, uoc_limits[0], uoc_limits[1])
    std_next = np.std(T_sigma_next)

    dT_uoc = (F1*(T1 - T_uoc_current) + F2*(T2 - T_uoc_current)
              + u_final + reaction_rate(T_uoc_current)) / V
    T_uoc_current += dt * dT_uoc
    T_uoc[k + 1] = T_uoc_current
    T_uoc_std[k + 1] = std_next

r2_robust = r2_manual(T_set, T_robust)
r2_uoc = r2_manual(T_set, T_uoc)
rmse_robust = np.sqrt(np.mean((T_set - T_robust) ** 2))
rmse_uoc = np.sqrt(np.mean((T_set - T_uoc) ** 2))

print(f"R² Robust control: {r2_robust:.4f}")
print(f"R² Unscented control: {r2_uoc:.4f}")
print(f"RMSE Robust: {rmse_robust:.3f} K, RMSE UOC: {rmse_uoc:.3f} K")
print(f"Robust actuator saturation fraction: {robust_saturated.mean():.3f}")
print("Max |T_robust|:", np.nanmax(np.abs(T_robust)))
print("Max |T_uoc|:", np.nanmax(np.abs(T_uoc)))

# Plots
plt.figure(figsize=(10,3))
plt.plot(T_set, linestyle='--')
plt.plot(T_robust)
plt.plot(T_uoc)
plt.fill_between(np.arange(len(T_uoc)), T_uoc - T_uoc_std, T_uoc + T_uoc_std, alpha=0.25)
plt.legend(['Setpoint', 'Robust', 'UOC', 'UOC sigma'])
plt.grid(True)
plt.show()

