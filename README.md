"""
thermo-active-VSA: System Identification Using the Kelvin-Voigt Model

This script simulates the mechanical behavior of a viscoelastic soft actuator
using the Kelvin-Voigt model. It generates synthetic strain and stress data,
adds noise to simulate experimental conditions, and applies a curve fitting 
algorithm to estimate model parameters (elastic modulus and viscosity).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# 1. Kelvin-Voigt Model
# =========================

def kelvin_voigt_model(epsilon, E1, eta, dt=0.1):
    """
    Computes the stress response for a given strain input using the Kelvin-Voigt model.

    Parameters:
    - epsilon (np.ndarray): Strain array as a function of time.
    - E1 (float): Elastic modulus (Pa).
    - eta (float): Viscosity coefficient (Pa·s).
    - dt (float): Time step size (s), used to calculate the time derivative.

    Returns:
    - sigma (np.ndarray): Computed stress (Pa) over time.
    
    Kelvin-Voigt constitutive equation:
    σ(t) = E1 * ε(t) + η * dε/dt
    """
    d_epsilon_dt = np.gradient(epsilon, dt)  # numerical derivative of strain
    sigma = E1 * epsilon + eta * d_epsilon_dt
    return sigma

# =========================
# 2. Generate Synthetic Strain Data
# =========================

t_max = 10            # Total simulation time in seconds
dt = 0.1              # Time step (s)
t = np.arange(0, t_max, dt)  # Time vector

# Simulated strain: exponentially rising function (e.g., actuation curve)
epsilon = 0.05 * (1 - np.exp(-0.5 * t))  # 5% strain asymptotically reached

# =========================
# 3. Ground Truth Parameters
# =========================

E1_true = 500  # True elastic modulus (Pa)
eta_true = 50  # True viscosity coefficient (Pa·s)

# Generate corresponding stress using the true parameters
sigma_true = kelvin_voigt_model(epsilon, E1_true, eta_true, dt)

# =========================
# 4. Add Noise to Simulate Real Experimental Data
# =========================

np.random.seed(0)  # For reproducibility
noise_std_dev = 5  # Standard deviation of noise in stress
sigma_noisy = sigma_true + np.random.normal(scale=noise_std_dev, size=len(sigma_true))

# =========================
# 5. System Identification via Least Squares Fitting
# =========================

def fit_kelvin_voigt(epsilon, E1, eta):
    """
    Wrapper function for curve fitting.
    Takes strain and parameters, returns predicted stress.
    """
    return kelvin_voigt_model(epsilon, E1, eta, dt)

# Initial guess for the parameters [E1, eta]
initial_guess = [100, 10]

# Perform the curve fit to estimate E1 and eta from noisy stress data
popt, _ = curve_fit(fit_kelvin_voigt, epsilon, sigma_noisy, p0=initial_guess)
E1_est, eta_est = popt  # Extract fitted parameters

# =========================
# 6. Generate Estimated Stress Response
# =========================

sigma_est = kelvin_voigt_model(epsilon, E1_est, eta_est, dt)

# =========================
# 7. Visualization of Results
# =========================

plt.figure(figsize=(10, 6))
plt.plot(t, sigma_true, label="True Stress (Ground Truth)", linestyle='dashed', color='black')
plt.plot(t, sigma_noisy, label="Noisy Observed Stress", alpha=0.6, color='red')
plt.plot(t, sigma_est, label=f"Estimated Stress\n(E1={E1_est:.2f} Pa, η={eta_est:.2f} Pa·s)", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Stress (Pa)")
plt.title("Kelvin-Voigt Model System Identification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 8. Save Results to CSV for Sharing
# =========================

output_data = np.column_stack((t, epsilon, sigma_noisy, sigma_est))
header = "Time(s),Strain,Observed_Stress(Pa),Estimated_Stress(Pa)"
np.savetxt("kelvin_voigt_results.csv", output_data, delimiter=",", header=header, comments='')

# =========================
# 9. Display Estimated Parameters
# =========================

print("=== Identified Parameters ===")
print(f"Estimated Elastic Modulus (E1): {E1_est:.2f} Pa")
print(f"Estimated Viscosity Coefficient (η): {eta_est:.2f} Pa·s")
