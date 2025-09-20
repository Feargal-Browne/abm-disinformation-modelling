"""
This module defines the PolicyInterventionSimulator class, which is used to
simulate various policy interventions in the agent-based model.
"""
import numpy as np
import pandas as pd
from enhanced_abm_framework import AdvancedAgentBasedModel
from enhanced_agent import EnhancedDisinformationAgent


class PolicyInterventionSimulator:
    """
    A class to simulate various policy interventions in the agent-based model.
    """

    def __init__(self, abm: AdvancedAgentBasedModel):
        """
        Initializes the PolicyInterventionSimulator.

        Args:
            abm (AdvancedAgentBasedModel): An instance of the AdvancedAgentBasedModel.
        """
        self.abm = abm

    def apply_fact_checking_campaign(self, time_step: int, target_agents_ids: list[int] = None,
                                     belief_reduction_factor: float = 0.5,
                                     beta_reduction_factor: float = 0.2) -> None:
        """
        Simulates a fact-checking campaign.
        Can target specific agents or apply broadly.
        Reduces belief and/or beta for affected agents.
        """
        print(f"Applying fact-checking campaign at time step {time_step}...")
        if target_agents_ids is None:
            # Apply to all agents (or a random subset)
            agents_to_affect = self.abm.agents
        else:
            agents_to_affect = [self.abm.agents[i]
                                for i in target_agents_ids]

        for agent_to_affect in agents_to_affect:
            # Rotate belief state towards skepticism
            rotation_angle = np.pi * belief_reduction_factor
            rotation_matrix = np.array([[np.cos(rotation_angle / 2), -np.sin(rotation_angle / 2)],
                                        [np.sin(rotation_angle / 2), np.cos(rotation_angle / 2)]])
            agent_to_affect.update_belief(rotation_matrix)

            # Reduce beta_i (propensity to spread) for agents who were spreading
            if agent_to_affect.measure_belief() == "Believer":
                agent_to_affect.base_beta *= (1.0 - beta_reduction_factor)

    def apply_media_literacy_program(self, time_step: int, target_agents_ids: list[int] = None,
                                     efficacy_increase: float = 0.2) -> None:
        """
        Simulates a media literacy program.
        Increases self-efficacy and response-efficacy for affected agents.
        """
        print(f"Applying media literacy program at time step {time_step}...")
        if target_agents_ids is None:
            agents_to_affect = self.abm.agents
        else:
            agents_to_affect = [self.abm.agents[i]
                                for i in target_agents_ids]

        for agent_to_affect in agents_to_affect:
            agent_to_affect.pmt['self_efficacy'] += efficacy_increase
            agent_to_affect.pmt['self_efficacy'] = np.clip(
                agent_to_affect.pmt['self_efficacy'], 0.0, 1.0)
            agent_to_affect.pmt['response_efficacy'] += efficacy_increase
            agent_to_affect.pmt['response_efficacy'] = np.clip(
                agent_to_affect.pmt['response_efficacy'], 0.0, 1.0)

    def apply_content_moderation(self, time_step: int, removal_rate: float = 0.1) -> None:
        """
        Simulates platform content moderation.
        Removes a percentage of 'Infected' agents from the active spreading pool.
        This could be modeled as setting their state to 'Restrained' or reducing their beta_i to zero.
        For simplicity, we'll set their beta_i to zero and change their status.
        """
        print(f"Applying content moderation at time step {time_step}...")
        infected_agents = [
            agent for agent in self.abm.agents if agent.measure_belief() == "Believer"]
        num_to_moderate = int(len(infected_agents) * removal_rate)

        # Randomly select agents to moderate
        if num_to_moderate > 0:
            agents_to_moderate = self.abm.rng.choice(
                infected_agents, num_to_moderate, replace=False)

            for agent_to_moderate in agents_to_moderate:
                agent_to_moderate.base_beta = 0.0  # Cannot spread anymore
                # Force belief state to skeptic
                agent_to_moderate.belief_state = np.array(
                    [1.0, 0.0], dtype=complex)

    def apply_public_awareness_campaign(self, time_step: int, target_agents_ids: list[int] = None,
                                        threat_increase: float = 0.2) -> None:
        """
        Simulates a public awareness campaign.
        Increases perceived severity and vulnerability for affected agents.
        """
        print(
            f"Applying public awareness campaign at time step {time_step}...")
        if target_agents_ids is None:
            agents_to_affect = self.abm.agents
        else:
            agents_to_affect = [self.abm.agents[i] for i in target_agents_ids]

        for agent_to_affect in agents_to_affect:
            agent_to_affect.pmt['severity'] += threat_increase
            agent_to_affect.pmt['severity'] = np.clip(
                agent_to_affect.pmt['severity'], 0.0, 1.0)
            agent_to_affect.pmt['vulnerability'] += threat_increase
            agent_to_affect.pmt['vulnerability'] = np.clip(
                agent_to_affect.pmt['vulnerability'], 0.0, 1.0)


if __name__ == "__main__":
    # --- Example Usage within a conceptual ABM simulation loop ---
    # This demonstrates how interventions would be called within the main simulation.

    # Dummy setup for ABM
    NUM_AGENTS_TEST = 100
    SEED_TEST = 42
    RNG_TEST = np.random.default_rng(SEED_TEST)

    # Dummy population data
    DUMMY_POP_DATA_SIZE = 100
    dummy_pop_latent_traits = RNG_TEST.normal(
        0, 1, size=(DUMMY_POP_DATA_SIZE, 5))
    dummy_pop_network_sizes = RNG_TEST.integers(10, 100, DUMMY_POP_DATA_SIZE)
    DUMMY_POPULATION_DATA = pd.DataFrame({
        'Openness_Score': dummy_pop_latent_traits[:, 0],
        'Conscientiousness_Score': dummy_pop_latent_traits[:, 1],
        'Extraversion_Score': dummy_pop_latent_traits[:, 2],
        'Agreeableness_Score': dummy_pop_latent_traits[:, 3],
        'Neuroticism_Score': dummy_pop_latent_traits[:, 4],
        'Network_Size': dummy_pop_network_sizes
    })

    abm_test = AdvancedAgentBasedModel(
        num_agents=NUM_AGENTS_TEST, population_data=DUMMY_POPULATION_DATA, seed=SEED_TEST)
    abm_test.initialize_agents()
    abm_test.build_network(m=3)

    # Initialize the PolicyInterventionSimulator
    policy_sim = PolicyInterventionSimulator(abm_test)

    # Conceptual simulation loop
    TIME_POINTS_TEST = np.linspace(0, 10, 11)  # 10 time steps
    Q_EXPONENT_TEST = 1.5
    MESSAGE_PROPERTIES_TEST = {
        'is_false': 0.8,
        'is_vague': 0.6,
        'is_irrelevant': 0.3
    }

    print("\n--- Running conceptual simulation with interventions ---")
    for t_idx, t in enumerate(TIME_POINTS_TEST):
        print(f"\nTime Step: {t_idx}")

        # Example: Apply fact-checking at time step 3
        if t_idx == 3:
            policy_sim.apply_fact_checking_campaign(
                time_step=t_idx, belief_reduction_factor=0.3)

        # Example: Apply media literacy program at time step 5 to a subset of agents
        if t_idx == 5:
            # Select 10 random agents to target
            target_agents = RNG_TEST.choice(
                abm_test.num_agents, 10, replace=False).tolist()
            policy_sim.apply_media_literacy_program(
                time_step=t_idx, target_agents_ids=target_agents, efficacy_increase=0.3)

        # Example: Apply content moderation at time step 7
        if t_idx == 7:
            policy_sim.apply_content_moderation(
                time_step=t_idx, removal_rate=0.05)

        # In a real ABM, here you would run the agent-level step function
        # that includes interactions, belief updates, and state transitions.
        # For this conceptual example, we just print some agent states.
        if t_idx == 0:
            # Initial state: seed some agents as 'Believer' for demonstration
            for i in range(5):
                abm_test.agents[i].belief_state = np.array(
                    [0.0, 1.0], dtype=complex)

        # Conceptual agent updates (replace with actual ABM step logic)
        num_infected = sum(
            1 for agent_in_loop in abm_test.agents if agent_in_loop.measure_belief() == "Believer")
        print(f"Number of Believers: {num_infected}")

    print("\nConceptual simulation with interventions completed.")
