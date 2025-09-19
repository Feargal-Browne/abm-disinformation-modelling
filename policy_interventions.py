import numpy as np
import pandas as pd
from enhanced_abm_framework import EnhancedAgentBasedModel
from enhanced_agent import DisinformationAgent

class PolicyInterventionSimulator:
    def __init__(self, abm: EnhancedAgentBasedModel):
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
            agents_to_affect = [self.abm.agents[i] for i in target_agents_ids]

        for agent in agents_to_affect:
            # Direct belief reduction
            agent.belief *= (1.0 - belief_reduction_factor)
            agent.belief = float(np.clip(agent.belief, 0.0, 1.0))
            agent.status = agent.update_status_from_belief()

            # Reduce beta_i (propensity to spread) for agents who were spreading
            if agent.status == "Infected": # Assuming 'Infected' means spreading
                agent.beta_i *= (1.0 - beta_reduction_factor)
                agent.beta_i = float(np.clip(agent.beta_i, 0.0, 1.0))

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
            agents_to_affect = [self.abm.agents[i] for i in target_agents_ids]

        for agent in agents_to_affect:
            agent.self_efficacy += efficacy_increase
            agent.self_efficacy = float(np.clip(agent.self_efficacy, 0.0, 1.0))
            agent.response_efficacy += efficacy_increase
            agent.response_efficacy = float(np.clip(agent.response_efficacy, 0.0, 1.0))

    def apply_content_moderation(self, time_step: int, removal_rate: float = 0.1) -> None:
        """
        Simulates platform content moderation.
        Removes a percentage of 'Infected' agents from the active spreading pool.
        This could be modeled as setting their state to 'Restrained' or reducing their beta_i to zero.
        For simplicity, we'll set their beta_i to zero and change their status.
        """
        print(f"Applying content moderation at time step {time_step}...")
        infected_agents = [agent for agent in self.abm.agents if agent.status == "Infected"]
        num_to_moderate = int(len(infected_agents) * removal_rate)
        
        # Randomly select agents to moderate
        agents_to_moderate = self.abm.rng.choice(infected_agents, num_to_moderate, replace=False)

        for agent in agents_to_moderate:
            agent.beta_i = 0.0 # Cannot spread anymore
            agent.status = "Restrained" # Change their status
            agent.belief = 0.0 # They no longer believe/spread

    def apply_public_awareness_campaign(self, time_step: int, target_agents_ids: list[int] = None,
                                        threat_increase: float = 0.2) -> None:
        """
        Simulates a public awareness campaign.
        Increases perceived severity and vulnerability for affected agents.
        """
        print(f"Applying public awareness campaign at time step {time_step}...")
        if target_agents_ids is None:
            agents_to_affect = self.abm.agents
        else:
            agents_to_affect = [self.abm.agents[i] for i in target_agents_ids]

        for agent in agents_to_affect:
            agent.perceived_severity += threat_increase
            agent.perceived_severity = float(np.clip(agent.perceived_severity, 0.0, 1.0))
            agent.perceived_vulnerability += threat_increase
            agent.perceived_vulnerability = float(np.clip(agent.perceived_vulnerability, 0.0, 1.0))


if __name__ == "__main__":
    # --- Example Usage within a conceptual ABM simulation loop ---
    # This demonstrates how interventions would be called within the main simulation.

    # Dummy setup for ABM
    num_agents_test = 100
    seed_test = 42
    rng_test = np.random.default_rng(seed_test)

    # Dummy population data
    dummy_pop_data_size = 100
    dummy_pop_latent_traits = rng_test.normal(0, 1, size=(dummy_pop_data_size, 5))
    dummy_pop_network_sizes = rng_test.integers(10, 100, dummy_pop_data_size)
    dummy_population_data = pd.DataFrame({
        'Openness_Score': dummy_pop_latent_traits[:, 0],
        'Conscientiousness_Score': dummy_pop_latent_traits[:, 1],
        'Extraversion_Score': dummy_pop_latent_traits[:, 2],
        'Agreeableness_Score': dummy_pop_latent_traits[:, 3],
        'Neuroticism_Score': dummy_pop_latent_traits[:, 4],
        'Network_Size': dummy_pop_network_sizes
    })

    abm_test = EnhancedAgentBasedModel(num_agents=num_agents_test, population_data=dummy_population_data, seed=seed_test)
    abm_test.initialize_agents()
    abm_test.build_fractal_network(m=3)

    # Initialize the PolicyInterventionSimulator
    policy_sim = PolicyInterventionSimulator(abm_test)

    # Conceptual simulation loop
    time_points_test = np.linspace(0, 10, 11) # 10 time steps
    q_exponent_test = 1.5
    message_properties_test = {
        'is_false': 0.8,
        'is_vague': 0.6,
        'is_irrelevant': 0.3
    }

    print("\n--- Running conceptual simulation with interventions ---")
    for t_idx, t in enumerate(time_points_test):
        print(f"\nTime Step: {t_idx}")

        # Example: Apply fact-checking at time step 3
        if t_idx == 3:
            policy_sim.apply_fact_checking_campaign(time_step=t_idx, belief_reduction_factor=0.3)

        # Example: Apply media literacy program at time step 5 to a subset of agents
        if t_idx == 5:
            # Select 10 random agents to target
            target_agents = rng_test.choice(abm_test.num_agents, 10, replace=False).tolist()
            policy_sim.apply_media_literacy_program(time_step=t_idx, target_agents_ids=target_agents, efficacy_increase=0.3)

        # Example: Apply content moderation at time step 7
        if t_idx == 7:
            policy_sim.apply_content_moderation(time_step=t_idx, removal_rate=0.05)

        # In a real ABM, here you would run the agent-level step function
        # that includes interactions, belief updates, and state transitions.
        # For this conceptual example, we just print some agent states.
        if t_idx == 0:
            # Initial state: seed some agents as 'Infected' for demonstration
            for i in range(5):
                abm_test.agents[i].belief = 0.9
                abm_test.agents[i].status = abm_test.agents[i].update_status_from_belief()

        # Conceptual agent updates (replace with actual ABM step logic)
        num_infected = sum(1 for agent in abm_test.agents if agent.status == "Infected")
        print(f"Number of Infected agents: {num_infected}")

        # Simulate some belief spread and decay for demonstration
        for agent in abm_test.agents:
            # Simulate interaction (very simplified)
            if agent.status == "Susceptible" and rng_test.random() < 0.1 * num_infected / num_agents_test:
                agent.belief += rng_test.random() * 0.2
                agent.status = agent.update_status_from_belief()
            agent.decay_belief()

    print("\nConceptual simulation with interventions completed.")


