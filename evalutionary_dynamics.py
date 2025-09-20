"""
This module defines the EvolutionaryDynamics class, which manages the
evolution of agent strategies based on payoffs.
"""
import numpy as np


class EvolutionaryDynamics:
    """ Manages the evolution of agent strategies based on payoffs. """

    def __init__(self, strategies: list, mutation_rate: float = 0.01):
        self.strategies = strategies
        self.mutation_rate = mutation_rate

    def evolve_strategies(self, agents: list, network_adj: np.ndarray, rng: np.random.Generator):
        """
        Agents update their strategies by imitating more successful neighbors.
        This is the core of the Evolutionary Game Theory (EGT) model.
        """
        new_strategies = [agent.strategy for agent in agents]

        for i, agent in enumerate(agents):
            neighbors = np.where(network_adj[i, :] > 0)[0]
            if len(neighbors) == 0:
                continue

            # Find the strategy of the most successful neighbor
            best_neighbor_strategy = None
            max_payoff = agent.payoff  # Start with own payoff

            for neighbor_idx in neighbors:
                if agents[neighbor_idx].payoff > max_payoff:
                    max_payoff = agents[neighbor_idx].payoff
                    best_neighbor_strategy = agents[neighbor_idx].strategy

            # Imitate if a better strategy was found
            if best_neighbor_strategy is not None:
                new_strategies[i] = best_neighbor_strategy

            # Randomly mutate strategy with a small probability
            if rng.random() < self.mutation_rate:
                new_strategies[i] = rng.choice(self.strategies)

        # Apply the new strategies and reset payoffs for the next round
        for i, agent in enumerate(agents):
            agent.strategy = new_strategies[i]
            agent.payoff = 0.0  # Reset payoff after evolution step
