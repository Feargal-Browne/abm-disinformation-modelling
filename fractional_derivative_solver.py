import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def fde_abm(alpha: float, f, y0: float, t_span: tuple, h: float):
    """
    Solves a Caputo fractional differential equation using the Adams-Bashforth-Moulton method.

    Args:
        alpha (float): The fractional order of the derivative (0 < alpha <= 1).
        f (callable): The function f(t, y) defining the FDE: D^alpha y(t) = f(t, y(t)).
        y0 (float): The initial condition y(0).
        t_span (tuple): A tuple (t_start, t_end) for the time interval.
        h (float): The step size for the numerical solution.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - t_vals: The time points.
               - y_vals: The numerical solution y(t) at those time points.
    """
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + h, h)
    n = len(t_vals)
    y_vals = np.zeros(n)
    y_vals[0] = y0

    # Store the history of f(t, y) values for memory effect
    f_history = np.zeros(n)

    for k in range(n - 1):
        # Store the current f value
        f_history[k] = f(t_vals[k], y_vals[k])

        # --- Predictor Step (Adams-Bashforth) ---
        # Summation term for the predictor
        predictor_sum = 0
        for j in range(k + 1):
            # Calculate predictor weights b_j
            b_j = (h**alpha / alpha) * ((k + 1 - j)**alpha - (k - j)**alpha)
            predictor_sum += b_j * f_history[j]

        # Calculate the predicted value y_p
        y_predicted = y0 + (1 / gamma(alpha)) * predictor_sum

        # Evaluate f at the predicted point
        f_predicted = f(t_vals[k+1], y_predicted)

        # --- Corrector Step (Adams-Moulton) ---
        # Summation term for the corrector
        corrector_sum = 0
        
        # Calculate corrector weight for the j=0 term
        a_0 = (h**alpha / (alpha * (alpha + 1))) * (k**alpha * (alpha - k + 1) + (k + 1)**alpha)
        corrector_sum += a_0 * f_history[0]

        # Calculate corrector weights for 1 <= j <= k
        for j in range(1, k + 1):
            a_j = (h**alpha / (alpha * (alpha + 1))) * (
                (k - j + 2)**(alpha + 1) + (k - j)**(alpha + 1) - 2 * (k - j + 1)**(alpha + 1)
            )
            corrector_sum += a_j * f_history[j]
        
        # The final corrected value is the solution for this step
        y_vals[k+1] = y0 + (1 / gamma(alpha)) * (
            (h**alpha / (alpha * (alpha + 1))) * f_predicted + corrector_sum
        )

    return t_vals, y_vals
    
if __name__ == "__main__":
    # --- 1. Define the FDE problem ---
    # The function f(t, y) for the equation D^alpha y(t) = -y(t)
    def f_relaxation(t, y):
        return -y

    # Initial condition
    y0 = 1
    
    # Time span and step size
    t_span = (0, 10)
    h = 0.05

    # --- 2. Solve for different fractional orders ---
    # Case 1: alpha = 1.0 (should match standard ODE solution e^-t)
    t1, y1 = fde_abm(alpha=1.0, f=f_relaxation, y0=y0, t_span=t_span, h=h)

    # Case 2: alpha = 0.85
    t2, y2 = fde_abm(alpha=0.85, f=f_relaxation, y0=y0, t_span=t_span, h=h)

    # Case 3: alpha = 0.5
    t3, y3 = fde_abm(alpha=0.5, f=f_relaxation, y0=y0, t_span=t_span, h=h)

    # --- 3. Plot the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t1, y1, label=r'$\alpha = 1.0$ (Classical Decay)', linewidth=2.5, color='red')
    ax.plot(t2, y2, label=r'$\alpha = 0.85$', linewidth=2, linestyle='--', color='blue')
    ax.plot(t3, y3, label=r'$\alpha = 0.50$', linewidth=2, linestyle=':', color='green')
    
    # For comparison, plot the true analytical solution for alpha=1.0
    ax.plot(t1, np.exp(-t1), label=r'Analytical $e^{-t}$', linestyle='-.', color='black', alpha=0.5)

    ax.set_title('Solution to the Fractional Relaxation Equation $D^\\alpha y(t) = -y(t)$', fontsize=14)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('y(t)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


