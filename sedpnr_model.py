"""
This module defines the SEDPNR (Susceptible, Exposed, Doubtful, Positively
Infected, Negatively Infected, Recovered) model, which is a system of
fractional differential equations used to simulate the spread of
disinformation.
"""
from typing import Tuple
import numpy as np
from fractional_derivative_solver import AdamsMoultonWeights


class SEDPNRModel:
    """
    The high-accuracy PECE solver, now adapted to handle dynamic inputs
    for fractional order (alpha) and agent transmission rates (beta).

    Note on naming conventions:
    - Variable names like S, E, D, P, N, R are used to match the standard
      notation for epidemiological compartments.
    - Attribute names like p_i_D_to_P are used to match the mathematical
      notation in the model's equations.
    """

    def __init__(self, num_agents, q_exponent, agent_params, network_adj_matrix, h, num_steps):
        self.num_agents = num_agents
        self.q_exponent = q_exponent
        self.network_adj_matrix = network_adj_matrix
        self.h = h

        # --- Static agent parameters ---
        self.sigma_i = np.array([p["sigma_i"]
                                for p in agent_params]).reshape(1, -1)
        self.rho_i = np.array([p["rho_i"]
                              for p in agent_params]).reshape(1, -1)
        self.p_i_D_to_P = np.array([p["p_i_D_to_P"]
                                   for p in agent_params]).reshape(1, -1)
        self.gamma_i_P_to_R = np.array(
            [p["gamma_i_P_to_R"] for p in agent_params]).reshape(1, -1)
        self.eta_i = np.array([p["eta_i"]
                              for p in agent_params]).reshape(1, -1)

        self.infectiousness_weights = np.array([0, 1.0, 0.5, 0.8, 0, 0])

        # --- Cache for PECE weights for different alphas ---
        self.weight_calculators: dict = {}
        self.num_steps = num_steps

    def get_weights_for_alpha(self, alpha: float) -> AdamsMoultonWeights:
        """ Lazily initializes and caches weight calculators for a given alpha. """
        if alpha not in self.weight_calculators:
            self.weight_calculators[alpha] = AdamsMoultonWeights(
                alpha, self.num_steps)
        return self.weight_calculators[alpha]

    def calculate_f(self, states: np.ndarray, dynamic_betas: np.ndarray) -> np.ndarray:
        """ Calculates f(t, y) using dynamic beta values from agent decisions. """
        # pylint: disable=unused-variable
        S, E, D, P, N, R = states
        F = self.infectiousness_weights @ states
        infection_matrix = self.network_adj_matrix * (F ** self.q_exponent)
        neighbor_infection_sum = infection_matrix.sum(axis=1).T
        force_of_infection = dynamic_betas * \
            (S ** self.q_exponent) * neighbor_infection_sum

        f = np.vstack([
            -force_of_infection, force_of_infection - self.sigma_i * E,
            self.sigma_i * E - self.rho_i * D,
            self.p_i_D_to_P * self.rho_i * D - self.gamma_i_P_to_R * P,
            (1 - self.p_i_D_to_P) * self.rho_i * D - self.eta_i * N,
            self.gamma_i_P_to_R * P + self.eta_i * N
        ])
        return f

    def step(self, t_idx: int, current_states: np.ndarray, f_history: np.ndarray,
             dynamic_betas: np.ndarray, current_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        """ Performs one PECE step with dynamic alpha and beta. """
        weights = self.get_weights_for_alpha(current_alpha)
        h_alpha = self.h ** current_alpha

        # --- PREDICTOR ---
        j_indices = np.arange(t_idx + 1)
        predictor_hist_sum = np.dot(
            f_history.T, weights.predictor_weights[j_indices])
        predicted_states = current_states + h_alpha * predictor_hist_sum.T

        # --- CORRECTOR ---
        f_predicted = self.calculate_f(predicted_states, dynamic_betas)
        j_indices_corr = np.arange(1, t_idx + 2)
        corrector_hist_sum = np.dot(
            f_history[1:].T, weights.corrector_weights[j_indices_corr[::-1][:-1]])

        next_states = current_states + h_alpha * (
            weights.corrector_weights[0] * f_predicted + corrector_hist_sum.T
        )

        next_states = np.clip(next_states, 0, 1)
        next_states /= next_states.sum(axis=0, keepdims=True)

        f_next = self.calculate_f(next_states, dynamic_betas)

        return next_states, f_next
