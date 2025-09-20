# enhanced_abm_framework.py
import numpy as np
import cupy as cp
import pandas as pd
from enhanced_agent import EnhancedDisinformationAgent
from sedpnr_model import SEDPNRModel
from platform_agent import PlatformAgent
from evolutionary_dynamics import EvolutionaryDynamics
import networkx as nx

class AdvancedAgentBasedModel:
    def __init__(self, num_agents=1000, population_data=None, seed=None):
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed)
        self.agents: list[EnhancedDisinformationAgent] = []
        self.network = None
        self.network_adj_matrix = None
        self.population_data = population_data

        # --- Initialize Advanced Components ---
        self.platforms = ['Facebook', 'Twitter'] # Example multiplex layers
        self.platform_agents = {p: PlatformAgent(p) for p in self.platforms}

        self.strategies = ['AggressiveSharer', 'CautiousVerifier']
        self.evolution_manager = EvolutionaryDynamics(self.strategies)

        self.h = 0.2 # Time step
        self.current_step = 0

    def initialize_agents(self):
        """ Initializes agents with strategies, PMT, and multiplex attributes. """
        print(f"Initializing {self.num_agents} advanced agents...")
        for i in range(self.num_agents):
            dummy_pmt = {'severity': self.rng.random(), 'vulnerability': self.rng.random(),
                         'self_efficacy': self.rng.random(), 'response_efficacy': self.rng.random()}
            initial_strategy = self.rng.choice(self.strategies)

            agent = EnhancedDisinformationAgent(
                agent_id=i, rng=self.rng, platforms=self.platforms,
                pmt_params=dummy_pmt, strategy=initial_strategy
            )
            agent.gamma = self.rng.uniform(0.01, 0.1)
            self.agents.append(agent)

    def build_network(self, m=5):
        """ Builds a simple scale-free network for demonstration. """
        print("Building agent network...")
        G = nx.barabasi_albert_graph(self.num_agents, m, seed=self.rng.integers(1e5))
        self.network = G
        self.network_adj_matrix = nx.to_numpy_array(G)

    def evolve_network(self):
        """
        Evolves the network structure based on agent interactions.
        Agents with similar beliefs are more likely to form connections.
        """
        for i in range(self.num_agents):
            agent1 = self.agents[i]
            # Randomly select another agent to potentially form a connection with
            j = self.rng.integers(self.num_agents)
            if i == j:
                continue
            agent2 = self.agents[j]

            # Probability of forming a connection is higher if beliefs are similar
            belief_similarity = 1 - abs(agent1.cognitive_module.belief - agent2.cognitive_module.belief)
            if self.rng.random() < belief_similarity * 0.1: # 10% chance if beliefs are identical
                self.network.add_edge(i, j)

        self.network_adj_matrix = nx.to_numpy_array(self.network)


    def run_simulation(self, sim_duration: float, q_exponent: float):
        time_points = np.arange(0, sim_duration, self.h)
        num_steps = len(time_points)

        # History buffers (on GPU)
        y_history = cp.zeros((num_steps, 6, self.num_agents))
        f_history = cp.zeros((num_steps, 6, self.num_agents))

        initial_states = np.zeros((self.num_agents, 6))
        initial_states[:, 0] = 1.0
        initial_states[:5, 0] = 0.0
        initial_states[:5, 1] = 1.0
        y_history[0, :, :] = cp.asarray(initial_states.T)

        static_params = [{'sigma_i': 0.1, 'rho_i': 0.1, 'p_i_D_to_P': 0.5,
                          'gamma_i_P_to_R': agent.gamma, 'eta_i': 0.05} for agent in self.agents]
        sedpnr_model = SEDPNRModel(self.num_agents, q_exponent, static_params, self.network_adj_matrix, self.h, num_steps)

        print("Running advanced simulation...")
        for t_idx in range(num_steps - 1):
            self.current_step = t_idx

            dynamic_betas = cp.zeros((1, self.num_agents))
            for platform in self.platforms:
                if t_idx % 50 == 0 and t_idx > 0:
                    self.platform_agents[platform].optimize_policy(self.num_agents, 50, 0.5)

                mod_level = self.platform_agents[platform].current_moderation_level

                for i, agent in enumerate(self.agents):
                    if agent.decide_to_share(platform, mod_level):
                        dynamic_betas[0, i] = 1.0
                        agent.cognitive_module.payoff += 0.1

            infected_proportion = cp.sum(y_history[t_idx, 3:5, :]) / self.num_agents
            current_alpha = float(cp.clip(0.95 - 0.2 * infected_proportion, 0.75, 0.95))

            if t_idx == 0:
                 f_history[0,:,:] = sedpnr_model._calculate_f_gpu(y_history[0,:,:], dynamic_betas)

            current_y = y_history[t_idx, :, :]
            current_f_history = f_history[:t_idx + 1, :, :]
            next_y, next_f = sedpnr_model.step(t_idx, current_y, current_f_history, dynamic_betas, current_alpha)
            y_history[t_idx + 1, :, :] = next_y
            f_history[t_idx + 1, :, :] = next_f

            for agent in self.agents:
                if agent.cognitive_module.payoff > 0:
                    agent.update_reputation(self.platforms[0], True)

            if t_idx % 100 == 0 and t_idx > 0:
                print(f"\n--- Evolving strategies at step {t_idx} ---")
                self.evolution_manager.evolve_strategies(self.agents, self.network_adj_matrix, self.rng)
                strat_counts = pd.Series([a.cognitive_module.strategy for a in self.agents]).value_counts()
                print(strat_counts)
                print("--- Evolution complete ---\n")

            # Evolve the network
            if t_idx % 20 == 0 and t_idx > 0:
                self.evolve_network()
                sedpnr_model.network_adj_matrix = cp.asarray(self.network_adj_matrix)


        print("Advanced simulation completed.")
        return cp.asnumpy(y_history).transpose((0, 2, 1))

if __name__ == '__main__':
    abm = AdvancedAgentBasedModel(num_agents=200, seed=42)
    abm.initialize_agents()
    abm.build_network()
    results = abm.run_simulation(sim_duration=10.0, q_exponent=1.2)
    print("\nSimulation finished. Result shape:", results.shape)


