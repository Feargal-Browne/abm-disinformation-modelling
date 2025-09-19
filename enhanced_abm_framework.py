import numpy as np
import pandas as pd
import networkx as nx
from enhanced_agent import DisinformationAgent # Use the enhanced agent
from sedpnr_model import SEDPNRModel # Assuming sedpnr_model.py is in the same directory
from ipc_regression import IPCRegression # Assuming ipc_regression.py is in the same directory
from network_generator import NetworkGenerator # Import the new network generator
from policy_interventions import PolicyInterventionSimulator # Import the policy intervention simulator

# --- IMPORTANT NOTE ON EXECUTION ENVIRONMENT ---
# Simulating 200,000 agents on a scale-free network with complex dynamics
# will be computationally very intensive. For actual execution, Google Colab
# or a dedicated high-performance computing environment is strongly recommended.
# ------------------------------------------------

class EnhancedAgentBasedModel:
    def __init__(self, num_agents=200000, population_data=None, seed=None):
        self.num_agents = num_agents
        self.rng = np.random.default_rng(seed) # Use a random number generator for reproducibility
        self.agents = [] # List to hold agent objects
        self.network = None # NetworkX graph (single layer for SEDPNR)
        self.network_adj_matrix = None # Adjacency matrix for SEDPNR model
        self.population_data = population_data # Data from survey, IRT, and IPC regression
        self.ipc_regressor = IPCRegression()
        self.current_step = 0
        self.network_generator = NetworkGenerator(self.rng)
        self.policy_simulator = PolicyInterventionSimulator(self) # Initialize policy simulator

    def initialize_agents(self):
        """
        Initializes agents with personality profiles, network sizes, and
        agent-specific SEDPNR parameters (beta_i, gamma_i, etc.).
        This assumes `self.population_data` contains the distributions or
        actual latent trait scores and network sizes from the survey.
        """
        print(f"Initializing {self.num_agents} agents...")

        if self.population_data is None:
            print("Warning: No population data provided. Generating dummy agent data.")
            # Generate dummy latent trait scores and network sizes for demonstration
            dummy_latent_traits = self.rng.normal(0, 1, size=(self.num_agents, 5)) # 5 Big 5 traits
            dummy_network_sizes = self.rng.integers(10, 1000, self.num_agents) # Random network sizes

            # Create a dummy DataFrame for IPC regression prediction
            agent_data_for_ipc = pd.DataFrame({
                # Ensure column names match those expected by IPCRegression
                'Openness_Score': dummy_latent_traits[:, 0],
                'Conscientiousness_Score': dummy_latent_traits[:, 1],
                'Extraversion_Score': dummy_latent_traits[:, 2],
                'Agreeableness_Score': dummy_latent_traits[:, 3],
                'Neuroticism_Score': dummy_latent_traits[:, 4],
                'Network_Size': dummy_network_sizes
            })

            # Fit dummy IPC models (in a real scenario, these would be pre-fitted)
            # Use a smaller dummy data for fitting IPC to avoid excessive computation during initialization
            dummy_ipc_data = self.ipc_regressor.generate_dummy_data(num_participants=1000)
            self.ipc_regressor.fit_ipc_models(dummy_ipc_data)

            # Predict agent-specific beta and gamma
            predicted_beta, predicted_gamma = self.ipc_regressor.predict_agent_parameters(agent_data_for_ipc)

            for i in range(self.num_agents):
                # For PMT, assume some mapping from latent traits or random for dummy data
                pmt_from_latent = {
                    'severity': self.rng.random(),
                    'vulnerability': self.rng.random(),
                    'self_efficacy': self.rng.random(),
                    'response_efficacy': self.rng.random()
                }
                self.agents.append(DisinformationAgent(
                    agent_id=i, rng=self.rng,
                    initial_belief=0.0,
                    ideology=dummy_latent_traits[i, 0], # Example: use Openness as proxy for ideology
                    pmt_from_latent=pmt_from_latent
                ))
                # Assign SEDPNR specific parameters to agent object for later use
                self.agents[i].beta_i = predicted_beta[i]
                self.agents[i].gamma_i = predicted_gamma[i]
                self.agents[i].sigma_i = self.rng.random() * 0.05 # Placeholder
                self.agents[i].rho_i = self.rng.random() * 0.05 # Placeholder
                self.agents[i].p_i_D_to_P = self.rng.random() # Placeholder
                self.agents[i].eta_i = self.rng.random() * 0.01 # Placeholder

        else:
            # In a real scenario, sample from self.population_data distributions
            # and use pre-fitted IPC regressor to assign parameters
            print("Using provided population data to initialize agents (conceptual)...")
            # This part would involve more sophisticated sampling and parameter assignment
            # based on the actual structure of `population_data`.
            # For now, it's a placeholder for how real data would be used.
            # Assume ipc_regressor is already fitted if population_data is provided
            if self.ipc_regressor.model_beta is None:
                print("Warning: IPC regressor not fitted. Fitting with dummy data.")
                dummy_ipc_data = self.ipc_regressor.generate_dummy_data(num_participants=1000)
                self.ipc_regressor.fit_ipc_models(dummy_ipc_data)

            for i in range(self.num_agents):
                # Example: randomly pick an agent from the population_data to assign properties
                # This is NOT proper sampling for 200k agents, but illustrates the data flow
                idx = self.rng.integers(0, len(self.population_data))
                agent_data = self.population_data.iloc[idx]

                # Assuming population_data has columns like 'Openness_Score', 'Network_Size', etc.
                agent_traits = agent_data[[
                    'Openness_Score', 'Conscientiousness_Score', 'Extraversion_Score',
                    'Agreeableness_Score', 'Neuroticism_Score'
                ]].values
                agent_network_size = agent_data['Network_Size']

                # Prepare data for IPC prediction for this single agent
                single_agent_ipc_data = pd.DataFrame({
                    'Openness_Score': [agent_traits[0]],
                    'Conscientiousness_Score': [agent_traits[1]],
                    'Extraversion_Score': [agent_traits[2]],
                    'Agreeableness_Score': [agent_traits[3]],
                    'Neuroticism_Score': [agent_traits[4]],
                    'Network_Size': [agent_network_size]
                })

                # Predict agent-specific beta and gamma using the pre-fitted IPC regressor
                predicted_beta, predicted_gamma = self.ipc_regressor.predict_agent_parameters(single_agent_ipc_data)

                pmt_from_latent = {
                    'severity': agent_data.get('Perceived_Severity', self.rng.random()),
                    'vulnerability': agent_data.get('Perceived_Vulnerability', self.rng.random()),
                    'self_efficacy': agent_data.get('Self_Efficacy', self.rng.random()),
                    'response_efficacy': agent_data.get('Response_Efficacy', self.rng.random())
                }

                self.agents.append(DisinformationAgent(
                    agent_id=i, rng=self.rng,
                    initial_belief=0.0,
                    ideology=agent_traits[0], # Example: use Openness as proxy for ideology
                    pmt_from_latent=pmt_from_latent
                ))
                # Assign SEDPNR specific parameters to agent object for later use
                self.agents[i].beta_i = predicted_beta[0]
                self.agents[i].gamma_i = predicted_gamma[0]
                self.agents[i].sigma_i = self.rng.random() * 0.05 # Placeholder
                self.agents[i].rho_i = self.rng.random() * 0.05 # Placeholder
                self.agents[i].p_i_D_to_P = self.rng.random() # Placeholder
                self.agents[i].eta_i = self.rng.random() * 0.01 # Placeholder

        print(f"Initialized {len(self.agents)} agents.")

    def build_fractal_network(self, m=3, num_communities=0, p_inter_community=0.0, rewire_prob=0.1, homophily_strength=0.5, multiplex_layers: list = None):
        """
        Builds a scale-free network for the agents using Barabasi-Albert model.
        Includes optional community structure and homophily rewiring.
        Can also generate a multiplex network.

        m: Number of edges to attach from a new node to existing nodes (for BA model).
        num_communities: If > 1, generates a network with community structure.
        p_inter_community: Probability of forming an edge between nodes in different communities.
        rewire_prob: Probability of rewiring an edge for homophily.
        homophily_strength: How strongly similar agents attract (lower means stronger attraction).
        multiplex_layers: List of dictionaries for multiplex network configuration.
        """
        agent_ideologies = np.array([agent.ideology for agent in self.agents])

        if multiplex_layers:
            print("Building multiplex network...")
            agent_attributes = {"ideology": agent_ideologies}
            multiplex_net = self.network_generator.generate_multiplex_network(
                num_nodes=self.num_agents, layer_configs=multiplex_layers, agent_attributes=agent_attributes
            )
            # For SEDPNR, we need a single aggregated network. This is a simplification.
            # A more complex approach would involve multi-layer diffusion.
            # For now, we'll aggregate by taking the union of all edges.
            self.network = nx.Graph()
            for layer_name, graph in multiplex_net.items():
                self.network.add_edges_from(graph.edges())
            print(f"Aggregated multiplex network: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges.")

        elif num_communities > 1:
            print("Building scale-free network with communities...")
            self.network = self.network_generator.generate_scale_free_with_communities(
                num_nodes=self.num_agents, m=m, num_communities=num_communities, 
                p_inter_community=p_inter_community, agent_ideologies=agent_ideologies
            )
        else:
            print("Building simple scale-free network...")
            self.network = nx.barabasi_albert_graph(self.num_agents, m, seed=self.rng.integers(0, 100000))

            print("Applying homophily rewiring to simple scale-free network...")
            edges_to_rewire = []
            for u, v in list(self.network.edges()): # Iterate over a copy as edges might be removed
                if self.rng.random() < rewire_prob:
                    if abs(agent_ideologies[u] - agent_ideologies[v]) > homophily_strength:
                        edges_to_rewire.append((u, v))

            for u, v in edges_to_rewire:
                if self.network.has_edge(u, v):
                    self.network.remove_edge(u, v)
                    potential_new_neighbors = [agent.id for agent in self.agents if 
                                               abs(agent_ideologies[u] - agent.ideologies[agent.id]) < homophily_strength and 
                                               not self.network.has_edge(u, agent.id) and 
                                               u != agent.id]
                    if potential_new_neighbors:
                        w = self.rng.choice(potential_new_neighbors)
                        self.network.add_edge(u, w)

        # Convert to adjacency matrix for SEDPNR model
        self.network_adj_matrix = nx.to_numpy_array(self.network)
        print(f"Network built with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges.")

    def run_simulation(self, time_points, q_exponent, message_properties: dict, 
                       interventions: list = None) -> np.ndarray:
        """
        Runs the SEDPNR simulation using the initialized agents and network.
        message_properties: Dictionary with 'is_false', 'is_vague', 'is_irrelevant' for IMT2.
        interventions: List of dictionaries, each defining an intervention:
            - 'time_step': When to apply the intervention.
            - 'type': Type of intervention (e.g., 'fact_checking', 'media_literacy', 'content_moderation', 'public_awareness').
            - 'params': Dictionary of parameters specific to the intervention type.
        """
        if not self.agents or self.network_adj_matrix is None:
            raise ValueError("Agents or network not initialized. Run `initialize_agents` and `build_fractal_network` first.")

        print("Preparing initial states for SEDPNR model...")
        # Initial states for SEDPNR model (e.g., all susceptible except a few exposed)
        initial_sedpnr_states = np.zeros((self.num_agents, 6))
        initial_sedpnr_states[:, 0] = 1.0 # All start as Susceptible

        # Seed a few agents as Exposed (e.g., first 5 agents)
        num_initial_exposed = min(5, self.num_agents)
        for i in range(num_initial_exposed):
            initial_sedpnr_states[i, 0] = 0.0 # Not Susceptible
            initial_sedpnr_states[i, 1] = 1.0 # Exposed

        # Prepare agent parameters for SEDPNRModel
        sedpnr_agent_params = []
        for agent in self.agents:
            sedpnr_agent_params.append({
                "beta_i": agent.beta_i,
                "gamma_i_P_to_R": agent.gamma_i,
                "sigma_i": agent.sigma_i,
                "rho_i": agent.rho_i,
                "p_i_D_to_P": agent.p_i_D_to_P,
                "eta_i": agent.eta_i,
            })

        print("Running SEDPNR simulation...")
        sedpnr_model = SEDPNRModel(
            num_agents=self.num_agents,
            q_exponent=q_exponent,
            initial_states=initial_sedpnr_states,
            agent_params=sedpnr_agent_params,
            network_adj_matrix=self.network_adj_matrix
        )

        # Dynamic persuasiveness from IMT2
        msg_is_false = message_properties.get('is_false', 0.5)
        msg_is_vague = message_properties.get('is_vague', 0.5)
        msg_is_irrelevant = message_properties.get('is_irrelevant', 0.5)
        # Example weighting for dynamic persuasiveness (can be calibrated)
        dynamic_persuasiveness_factor = (msg_is_vague * 0.5) + (msg_is_false * 0.2) + (msg_is_irrelevant * 0.1)
        dynamic_persuasiveness_factor = float(np.clip(dynamic_persuasiveness_factor, 0.0, 1.0))

        # Conceptual simulation loop to show where new features fit
        simulation_results = []
        current_sedpnr_states = initial_sedpnr_states.copy()

        for t_idx, t in enumerate(time_points):
            self.current_step = t_idx
            
            # Apply interventions at the specified time step
            if interventions:
                for intervention in interventions:
                    if intervention['time_step'] == t_idx:
                        interv_type = intervention['type']
                        interv_params = intervention.get('params', {})
                        print(f"Applying {interv_type} intervention at time step {t_idx}...")
                        if interv_type == 'fact_checking':
                            self.policy_simulator.apply_fact_checking_campaign(**interv_params)
                        elif interv_type == 'media_literacy':
                            self.policy_simulator.apply_media_literacy_program(**interv_params)
                        elif interv_type == 'content_moderation':
                            self.policy_simulator.apply_content_moderation(**interv_params)
                        elif interv_type == 'public_awareness':
                            self.policy_simulator.apply_public_awareness_campaign(**interv_params)
                        else:
                            print(f"Warning: Unknown intervention type: {interv_type}")

            # --- Agent-level interactions and updates (conceptual integration) ---
            # In a true ABM, this would be the core loop.
            # For each agent, identify neighbors, calculate interactions, update belief.
            # Then, update SEDPNR states based on agent beliefs.

            # For now, we'll use the SEDPNRModel's ODE solver for state transitions
            # and assume the agent-level dynamics influence the parameters passed to it.
            # This is a simplification until a full step-by-step ABM is implemented.

            # This part needs to be replaced by a proper agent-level step function
            # that updates each agent's S,E,D,P,N,R state based on their belief and interactions.
            # The SEDPNRModel's _ode_system would then be used to calculate rates of change
            # for each agent's compartment based on their individual parameters and network state.

            # For now, we'll just run the SEDPNRModel as is, and the agent-level features
            # will be implicitly represented by the initial agent parameters and the conceptual
            # dynamic_persuasiveness_factor.

            # This is a placeholder for the actual agent-level simulation step.
            # It would involve iterating through agents, their neighbors, and updating their states.
            # The `sedpnr_model._ode_system` would be called for each agent or for the whole system
            # with dynamically updated parameters based on agent interactions.

            # To bridge the gap, let's assume the SEDPNRModel can take an updated set of agent_params
            # that reflect the current state of beliefs and interactions.
            # This is still a conceptual leap without a full step-by-step ABM implementation.

            # For the purpose of this file, we'll simulate the SEDPNR model directly,
            # acknowledging the agent-level details are conceptual for now.
            # The `sedpnr_model.simulate` method is designed for batch ODE solving.
            # A true ABM would have a `step` method that updates agent states one by one.

            if t_idx == 0:
                solution = sedpnr_model.simulate(time_points)
            simulation_results.append(solution[t_idx, :, :]) # Append current time step's states

        print("SEDPNR simulation completed.")
        return np.array(simulation_results)


if __name__ == "__main__":
    # --- Conceptual Workflow ---
    # 1. Load/Generate population data (from survey, IRT, IPC regression)
    #    For this example, we'll use dummy data.
    dummy_pop_data_size = 1000
    dummy_pop_latent_traits = np.random.default_rng(42).normal(0, 1, size=(dummy_pop_data_size, 5))
    dummy_pop_network_sizes = np.random.default_rng(42).integers(10, 1000, dummy_pop_data_size)
    dummy_population_data = pd.DataFrame({
        'Openness_Score': dummy_pop_latent_traits[:, 0],
        'Conscientiousness_Score': dummy_pop_latent_traits[:, 1],
        'Extraversion_Score': dummy_pop_latent_traits[:, 2],
        'Agreeableness_Score': dummy_pop_latent_traits[:, 3],
        'Neuroticism_Score': dummy_pop_latent_traits[:, 4],
        'Network_Size': dummy_pop_network_sizes
    })

    # Initialize ABM
    # Use a smaller num_agents for quick test during development
    abm = EnhancedAgentBasedModel(num_agents=100, population_data=dummy_population_data, seed=42)

    # Initialize agents (assigns parameters based on population data)
    abm.initialize_agents()

    # Build the fractal network with homophily and communities/multiplex
    # Example 1: Simple scale-free with homophily
    # abm.build_fractal_network(m=3, rewire_prob=0.1, homophily_strength=0.5)

    # Example 2: Scale-free with communities and homophily
    abm.build_fractal_network(m=3, num_communities=5, p_inter_community=0.01, rewire_prob=0.1, homophily_strength=0.5)

    # Example 3: Multiplex network (will aggregate to a single network for SEDPNR)
    # layer_configs = [
    #     {"name": "Facebook", "type": "scale_free", "params": {"m": 5, "apply_homophily": True, "rewire_prob": 0.1, "homophily_strength": 0.5}},
    #     {"name": "Twitter", "type": "random", "params": {"p": 0.005, "apply_homophily": True, "rewire_prob": 0.2, "homophily_strength": 0.3}}
    # ]
    # abm.build_fractal_network(multiplex_layers=layer_configs)

    # Define time points for simulation
    time_points = np.linspace(0, 50, 51) # Simulate for 50 time units

    # Define message properties for IMT2
    message_properties = {
        'is_false': 0.8,
        'is_vague': 0.6,
        'is_irrelevant': 0.3
    }

    # Define policy interventions
    interventions_list = [
        {'time_step': 10, 'type': 'fact_checking', 'params': {'belief_reduction_factor': 0.3}},
        {'time_step': 25, 'type': 'media_literacy', 'params': {'efficacy_increase': 0.2}},
        {'time_step': 40, 'type': 'content_moderation', 'params': {'removal_rate': 0.1}}
    ]

    # Run the simulation
    q_exponent = 1.5 # Example Tsallis exponent for super-linear spread
    simulation_results = abm.run_simulation(time_points, q_exponent, message_properties, interventions=interventions_list)

    print("\nABM Simulation Results Shape:", simulation_results.shape)
    print("First few time steps of total Susceptible population:")
    print(simulation_results[:, :, 0].sum(axis=1)[:5])

    # --- Next Steps / Considerations ---
    print("\n--- Next Steps / Considerations ---")
    print("1. **Full Agent-Level Step Function:** The current `run_simulation` conceptually integrates agent-level dynamics. A dedicated `step` method within the ABM class that iterates through agents and applies their `interact_update`, `decay_belief`, and `update_status_from_belief` methods is needed for a true agent-based simulation. This would then feed into the SEDPNR state transitions.")
    print("2. **Fractional Derivatives:** The `SEDPNRModel` currently uses standard ODEs. Integration of a proper FDE solver is crucial for the fractal aspect. This will likely require a custom numerical solver or a specialized library that can handle fractional derivatives in a step-by-step manner for each agent.")
    print("3. **Model Calibration:** The `w_E, w_D, w_P` in `SEDPNRModel` and `m` in `build_fractal_network` need to be calibrated using real-world data and a Genetic Algorithm (GA) with Dynamic Time Warping (DTW) as described in the plan.")
    print("4. **Output Analysis:** Develop tools to analyze and visualize the simulation results (e.g., spread curves, state distributions over time).")
    print("5. **Policy Interventions:** Design mechanisms within the ABM to simulate policy interventions and evaluate their impact on disinformation spread.")


