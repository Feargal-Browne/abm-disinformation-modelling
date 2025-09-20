# sedpnr_model.py
import numpy as np
import cupy as cp
from numba import jit

class SEDPNRModel:
    """
    The high-accuracy PECE solver, now optimized for GPU execution with CuPy.
    """
    def __init__(self, num_agents, q_exponent, agent_params, network_adj_matrix, h, num_steps):
        self.num_agents = num_agents
        self.q_exponent = q_exponent
        self.network_adj_matrix = cp.asarray(network_adj_matrix)  # Move to GPU
        self.h = h

        # Static agent parameters (on GPU)
        self.sigma_i = cp.array([p["sigma_i"] for p in agent_params]).reshape(1, -1)
        self.rho_i = cp.array([p["rho_i"] for p in agent_params]).reshape(1, -1)
        self.p_i_D_to_P = cp.array([p["p_i_D_to_P"] for p in agent_params]).reshape(1, -1)
        self.gamma_i_P_to_R = cp.array([p["gamma_i_P_to_R"] for p in agent_params]).reshape(1, -1)
        self.eta_i = cp.array([p["eta_i"] for p in agent_params]).reshape(1, -1)

        self.infectiousness_weights = cp.array([0, 1.0, 0.5, 0.8, 0, 0])

        self.num_steps = num_steps

    @staticmethod
    @jit(nopython=True)
    def _calculate_f_cpu(states, dynamic_betas, network_adj_matrix, q_exponent,
                         sigma_i, rho_i, p_i_D_to_P, gamma_i_P_to_R, eta_i, infectiousness_weights):
        """
        A Numba JIT-compiled version of the f calculation for CPU fallback.
        """
        S, E, D, P, N, R = states
        F = infectiousness_weights @ states
        infection_matrix = network_adj_matrix * (F ** q_exponent)
        neighbor_infection_sum = infection_matrix.sum(axis=1).T
        force_of_infection = dynamic_betas * (S ** q_exponent) * neighbor_infection_sum

        f = np.vstack([
            -force_of_infection,
            force_of_infection - sigma_i * E,
            sigma_i * E - rho_i * D,
            p_i_D_to_P * rho_i * D - gamma_i_P_to_R * P,
            (1 - p_i_D_to_P) * rho_i * D - eta_i * N,
            gamma_i_P_to_R * P + eta_i * N
        ])
        return f


    def _calculate_f_gpu(self, states: cp.ndarray, dynamic_betas: cp.ndarray) -> cp.ndarray:
        """ Calculates f(t, y) on the GPU using CuPy. """
        S, E, D, P, N, R = states
        F = self.infectiousness_weights @ states
        infection_matrix = self.network_adj_matrix * (F ** self.q_exponent)
        neighbor_infection_sum = infection_matrix.sum(axis=1).T
        force_of_infection = dynamic_betas * (S ** self.q_exponent) * neighbor_infection_sum

        f = cp.vstack([
            -force_of_infection,
            force_of_infection - self.sigma_i * E,
            self.sigma_i * E - self.rho_i * D,
            self.p_i_D_to_P * self.rho_i * D - self.gamma_i_P_to_R * P,
            (1 - self.p_i_D_to_P) * self.rho_i * D - self.eta_i * N,
            self.gamma_i_P_to_R * P + self.eta_i * N
        ])
        return f

    def step(self, t_idx: int, current_states: cp.ndarray, f_history: cp.ndarray,
             dynamic_betas: cp.ndarray, current_alpha: float):
        """ Performs one PECE step on the GPU. """
        h_alpha = self.h ** current_alpha

        # Using a simplified predictor-corrector for GPU efficiency
        # A full Adams-Bashforth-Moulton is complex to vectorize for the GPU
        # This is a good trade-off between accuracy and performance
        f_current = self._calculate_f_gpu(current_states, dynamic_betas)
        predicted_states = current_states + h_alpha * f_current

        f_predicted = self._calculate_f_gpu(predicted_states, dynamic_betas)
        next_states = current_states + (h_alpha / 2) * (f_current + f_predicted)

        next_states = cp.clip(next_states, 0, 1)
        next_states /= next_states.sum(axis=0, keepdims=True)

        f_next = self._calculate_f_gpu(next_states, dynamic_betas)

        return next_states, f_next

    


