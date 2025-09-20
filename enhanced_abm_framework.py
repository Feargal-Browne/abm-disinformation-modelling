"""
This module defines the AdvancedAgentBasedModel class, which is a more
advanced version of the agent-based model. It incorporates features such as
multiplex networks, evolutionary dynamics, and platform agents.
"""
import networkx as nx
import numpy as np
import pandas as pd
from enhanced_agent import EnhancedDisinformationAgent
from sedpnr_model import SEDPNRModel
from platform_agent import PlatformAgent
from evolutionary_dynamics import EvolutionaryDynamics


class AdvancedAgentBasedModel:
    """
    A more advanced version of the agent-based model.
    """

    def __init__(self, num_agents=1000, population_data=None, seed=None):
        """
        Initializes the AdvancedAgentBasedModel.

        Args:
            num_agents (int): The number of agents in the simulation.
            population_data (pd.DataFrame, optional): The population data for the agents. Defaults to None.
            seed (int, optional): The random seed for reproducibility. Defaults to None.
        """
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)
        self.agents: list[EnhancedDisinformationAgent] = []
        self.network = None
        self.network_adj_matrix = None
        self.population_data = population_data

        # --- Initialize Advanced Components ---
        self.platforms = ['Facebook', 'Twitter']  # Example multiplex layers
        self.platform_agents = {
            p: PlatformAgent(p) for p in self.platforms}

        self.strategies = ['AggressiveSharer', 'CautiousVerifier']
        self.evolution_manager = EvolutionaryDynamics(self.strategies)

        self.h = 0.2  # Time step
        self.current_step = 0

    def initialize_agents(self):
        """ Initializes agents with strategies, PMT, and multiplex attributes. """
        print(f"Initializing {self.num_agents} advanced agents...")
        for i in range(self.num_agents):
            # In a real run, PMT params would come from population_data
            dummy_pmt = {'severity': self.rng.random(), 'vulnerability': self.rng.random(),
                         'self_efficacy': self.rng.random(), 'response_efficacy': self.rng.random()}
            initial_strategy = self.rng.choice(self.strategies)

            agent = EnhancedDisinformationAgent(
                agent_id=i, rng=self.rng, platforms=self.platforms,
                pmt_params=dummy_pmt, strategy=initial_strategy
            )
            # IPC regression would set these base values
            agent.gamma = self.rng.uniform(0.01, 0.1)
            self.agents.append(agent)

    def build_network(self, m=5):
        """ Builds a simple scale-free network for demonstration. """
        print("Building agent network...")
        # A full implementation would use the multiplex generator
        graph = nx.barabasi_albert_graph(
            self.num_agents, m, seed=self.rng.integers(1e5))
        self.network_adj_matrix = nx.to_numpy_array(graph)

    def run_simulation(self, sim_duration: float, q_exponent: float):
        """
        Runs the advanced simulation.

        Args:
            sim_duration (float): The duration of the simulation.
            q_exponent (float): The q-exponent for the Tsallis statistics.
        """
        time_points = np.arange(0, sim_duration, self.h)
        num_steps = len(time_points)

        # --- History Buffers ---
        y_history = np.zeros((num_steps, 6, self.num_agents))
        f_history = np.zeros((num_steps, 6, self.num_agents))

        # Set initial SEDPNR states
        initial_states = np.zeros((self.num_agents, 6))
        initial_states[:, 0] = 1.0
        initial_states[:5, 0] = 0.0
        initial_states[:5, 1] = 1.0
        y_history[0, :, :] = initial_states.T

        # Initialize SEDPNR model
        # Static params are now simpler as beta is dynamic
        static_params = [{'sigma_i': 0.1, 'rho_i': 0.1, 'p_i_D_to_P': 0.5,
                          'gamma_i_P_to_R': agent.gamma, 'eta_i': 0.05} for agent in self.agents]
        sedpnr_model = SEDPNRModel(
            self.num_agents, q_exponent, static_params, self.network_adj_matrix, self.h, num_steps)

        print("Running advanced simulation...")
        for t_idx in range(num_steps - 1):
            self.current_step = t_idx

            # --- 1. Agent Decision & Platform Policy Phase ---
            dynamic_betas = np.zeros((1, self.num_agents))
            for platform in self.platforms:
                # Platform updates its policy periodically
                if t_idx % 50 == 0 and t_idx > 0:
                    # These would be real metrics from the simulation
                    self.platform_agents[platform].optimize_policy(
                        self.num_agents, 50, 0.5)

                mod_level = self.platform_agents[platform].current_moderation_level

                for i, agent in enumerate(self.agents):
                    if agent.decide_to_share(platform, mod_level):
                        # For simplicity, sharing on any platform makes beta=1 for the single-layer SEDPNR
                        # This would be more nuanced in a full multiplex SEDPNR
                        dynamic_betas[0, i] = 1.0
                        # Successful share increases payoff
                        agent.payoff += 0.1

            # --- 2. Meta-Adaptation: Dynamic Alpha ---
            # Memory deepens as more agents become infected (P or N)
            infected_proportion = np.sum(
                y_history[t_idx, 3:5, :]) / self.num_agents
            current_alpha = np.clip(
                0.95 - 0.2 * infected_proportion, 0.75, 0.95)

            # --- 3. SEDPNR Propagation Phase ---
            if t_idx == 0:
                f_history[0, :, :] = sedpnr_model.calculate_f(
                    y_history[0, :, :], dynamic_betas)

            current_y = y_history[t_idx, :, :]
            current_f_history = f_history[:t_idx + 1, :, :]
            next_y, next_f = sedpnr_model.step(
                t_idx, current_y, current_f_history, dynamic_betas, current_alpha)
            y_history[t_idx + 1, :, :] = next_y
            f_history[t_idx + 1, :, :] = next_f

            # --- 4. Outcome & Update Phase (Simplified) ---
            # A full implementation would have logic to determine successful shares
            # and update agent reputations accordingly.
            for agent in self.agents:
                if agent.payoff > 0:  # Simple check if they shared
                    # Assume success for now
                    agent.update_reputation(self.platforms[0], True)

            # --- 5. Evolutionary Dynamics Phase ---
            if t_idx % 100 == 0 and t_idx > 0:
                print(f"\n--- Evolving strategies at step {t_idx} ---")
                self.evolution_manager.evolve_strategies(
                    self.agents, self.network_adj_matrix, self.rng)
                strat_counts = pd.Series(
                    [a.strategy for a in self.agents]).value_counts()
                print(strat_counts)
                print("--- Evolution complete ---\n")

        print("Advanced simulation completed.")
        return np.transpose(y_history, (0, 2, 1))


if __name__ == '__main__':
    # This block demonstrates how to run the fully integrated model
    abm = AdvancedAgentBasedModel(num_agents=200, seed=42)
    abm.initialize_agents()
    abm.build_network()

    results = abm.run_simulation(sim_duration=10.0, q_exponent=1.2)

    print("\nSimulation finished. Result shape:", results.shape)
