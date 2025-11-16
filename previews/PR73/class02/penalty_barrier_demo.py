
import numpy as np
import matplotlib.pyplot as plt

def plot_log_barrier(save_path="log_barrier.png"):
    x = np.linspace(1e-4, 1.0, 2000)
    y = -np.log(x)
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.xlabel("x")
    plt.ylabel("-log(x)")
    plt.title("Log barrier: -log(x) vs x (blows up as x → 0⁺)")
    plt.axvline(0.0, linestyle="--")
    plt.ylim(0, min(15, float(np.max(y))))
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def plot_quadratic_penalty(rhos=(1,10,100,1000,2500,5000), save_path="quadratic_penalty.png"):
    x = np.linspace(-1.0, 1.0, 2000)
    plt.figure()
    for rho in rhos:
        penalty = 0.5 * rho * np.minimum(0.0, x) ** 2
        plt.plot(x, penalty, linewidth=2, label=f"rho = {rho}")
    plt.xlabel("x (feasible region is x ≥ 0)")
    plt.ylabel("penalty(x) = (ρ/2)·min(0,x)²")
    plt.title("Quadratic penalty for inequality constraint vs. ρ")
    plt.axvline(0.0, linestyle="--")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_log_barrier()
    plot_quadratic_penalty()