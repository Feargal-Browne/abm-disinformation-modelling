import numpy as np
import pandas as pd
import networkx as nx
from sedpnr_model import SEDPNRModel # Assuming sedpnr_model.py is in the same directory
from ipc_regression import IPCRegression # Assuming ipc_regression.py is in the same directory

# --- IMPORTANT NOTE ON EXECUTION ENVIRONMENT ---
# Simulating 200,000 agents on a scale-free network with complex dynamics
# will be computationally very intensive. For actual execution, Google Colab
# or a dedicated high-performance computing environment is strongly recommended.
# ------------------------------------------------

class AgentBasedModel:
    def __init__(self, num_agents=200000, population_data=None):
        self.num_agents = num_agents
        self.agents = [] # List to hold agent objects/dictionaries
        self.network = None
        self.population_data = population_data # Data from survey, IRT, and IPC regression
        self.ipc_regressor = IPCRegression()

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
            # In a real scenario, these would be sampled from distributions derived from survey data
            dummy_latent_traits = np.random.randn(self.num_agents, 5) # 5 Big 5 traits
            dummy_network_sizes = np.random.randint(10, 1000, self.num_agents) # Random network sizes

            # Create a dummy DataFrame for IPC regression prediction
            agent_data_for_ipc = pd.DataFrame({
                'Openness_Score': dummy_latent_traits[:, 0],
                'Conscientiousness_Score': dummy_latent_traits[:, 1],
                'Extraversion_Score': dummy_latent_traits[:, 2],
                'Agreeableness_Score': dummy_latent_traits[:, 3],
                'Neuroticism_Score': dummy_latent_traits[:, 4],
                'Network_Size': dummy_network_sizes
            })

            # Fit dummy IPC models (in a real scenario, these would be pre-fitted)
            dummy_ipc_data = self.ipc_regressor.generate_dummy_data(num_participants=1000) # Use smaller dummy data for fitting
            self.ipc_regressor.fit_ipc_models(dummy_ipc_data)

            # Predict agent-specific beta and gamma
            predicted_beta, predicted_gamma = self.ipc_regressor.predict_agent_parameters(agent_data_for_ipc)

            for i in range(self.num_agents):
                self.agents.append({
                    "id": i,
                    "personality_scores": dummy_latent_traits[i, :],
                    "network_size": dummy_network_sizes[i],
                    "beta_i": predicted_beta[i],
                    "gamma_i": predicted_gamma[i],
                    "sigma_i": np.random.rand() * 0.05, # Placeholder
                    "rho_i": np.random.rand() * 0.05, # Placeholder
                    "p_i_D_to_P": np.random.rand(), # Placeholder
                    "eta_i": np.random.rand() * 0.01 # Placeholder
                })
        else:
            # In a real scenario, sample from self.population_data distributions
            # and use pre-fitted IPC regressor to assign parameters
            print("Using provided population data to initialize agents (conceptual)...")
            # This part would involve more sophisticated sampling and parameter assignment
            # based on the actual structure of `population_data`.
            # For now, it's a placeholder for how real data would be used.
            for i in range(self.num_agents):
                # Example: randomly pick an agent from the population_data to assign properties
                # This is NOT proper sampling for 200k agents, but illustrates the data flow
                idx = np.random.randint(0, len(self.population_data))
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

                self.agents.append({
                    "id": i,
                    "personality_scores": agent_traits,
                    "network_size": agent_network_size,
                    "beta_i": predicted_beta[0],
                    "gamma_i": predicted_gamma[0],
                    "sigma_i": np.random.rand() * 0.05, # Placeholder
                    "rho_i": np.random.rand() * 0.05, # Placeholder
                    "p_i_D_to_P": np.random.rand(), # Placeholder
                    "eta_i": np.random.rand() * 0.01 # Placeholder
                })

        print(f"Initialized {len(self.agents)} agents.")

    def build_fractal_network(self):
        """
        Builds a scale-free network for the agents.
        This uses the Barabasi-Albert model as a starting point for scale-free properties.
        Further enhancements for community structure and multiplexing would be added here.
        """
        print("Building scale-free network...")
        # Use Barabasi-Albert model for scale-free property
        # m: Number of edges to attach from a new node to existing nodes
        # A typical value for social networks is around 2-5.
        # The exact value of m can be calibrated.
        m = 3 # Example value
        self.network = nx.barabasi_albert_graph(self.num_agents, m)

        # Assign edge weights based on agent properties (e.g., network_size, tie strength)
        # This is a conceptual assignment. A more sophisticated method would be needed.
        adj_matrix = nx.to_numpy_array(self.network)
        for i, j in self.network.edges():
            # Example: edge weight influenced by the network size of the connected agents
            weight = (self.agents[i]["network_size"] + self.agents[j]["network_size"]) / (2 * max(self.agents[k]["network_size"] for k in range(self.num_agents)))
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight # Ensure symmetry for undirected graph

        self.network_adj_matrix = adj_matrix
        print(f"Network built with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges.")

    def run_simulation(self, time_points, q_exponent):
        """
        Runs the SEDPNR simulation using the initialized agents and network.
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
                "beta_i": agent["beta_i"],
                "gamma_i_P_to_R": agent["gamma_i"],
                "sigma_i": agent["sigma_i"],
                "rho_i": agent["rho_i"],
                "p_i_D_to_P": agent["p_i_D_to_P"],
                "eta_i": agent["eta_i"],
            })

        print("Running SEDPNR simulation...")
        sedpnr_model = SEDPNRModel(
            num_agents=self.num_agents,
            q_exponent=q_exponent,
            initial_states=initial_sedpnr_states,
            agent_params=sedpnr_agent_params,
            network_adj_matrix=self.network_adj_matrix
        )

        solution = sedpnr_model.simulate(time_points)
        print("SEDPNR simulation completed.")
        return solution


if __name__ == "__main__":
    # --- Conceptual Workflow ---
    # 1. Load/Generate population data (from survey, IRT, IPC regression)
    #    For this example, we'll use dummy data.
    #    In a real scenario, `population_data` would be a DataFrame containing
    #    latent trait scores, network sizes, and potentially other derived properties
    #    for the 1000 survey participants.
    #    The ABM will then sample from this data to create 200,000 agents.

    # Dummy population data (e.g., from 1000 survey participants after IRT/IPC)
    # This would be the output of `ipc_regression.py`'s `generate_dummy_data`
    # if it were run with 1000 participants and then used to train the IPC models.
    # For simplicity, let's create a small dummy population data here.
    dummy_pop_data_size = 1000
    dummy_pop_latent_traits = np.random.randn(dummy_pop_data_size, 5)
    dummy_pop_network_sizes = np.random.randint(10, 1000, dummy_pop_data_size)
    dummy_population_data = pd.DataFrame({
        'Openness_Score': dummy_pop_latent_traits[:, 0],
        'Conscientiousness_Score': dummy_pop_latent_traits[:, 1],
        'Extraversion_Score': dummy_pop_latent_traits[:, 2],
        'Agreeableness_Score': dummy_pop_latent_traits[:, 3],
        'Neuroticism_Score': dummy_pop_latent_traits[:, 4],
        'Network_Size': dummy_pop_network_sizes
    })

    # Initialize ABM
    abm = AgentBasedModel(num_agents=1000, population_data=dummy_population_data) # Use smaller num_agents for quick test

    # Initialize agents (assigns parameters based on population data)
    abm.initialize_agents()

    # Build the fractal network
    abm.build_fractal_network()

    # Define time points for simulation
    time_points = np.linspace(0, 50, 51) # Simulate for 50 time units

    # Run the simulation
    q_exponent = 0.5 # Example Tsallis exponent
    simulation_results = abm.run_simulation(time_points, q_exponent)

    print("\nABM Simulation Results Shape:", simulation_results.shape)
    print("First few time steps of total Susceptible population:")
    print(simulation_results[:, :, 0].sum(axis=1)[:5])

    # --- Next Steps / Considerations ---
    print("\n--- Next Steps / Considerations ---")
    print("1. **Fractional Derivatives:** The `SEDPNRModel` currently uses standard ODEs. Integration of a proper FDE solver is crucial for the fractal aspect.")
    print("2. **Network Refinement:** Implement more sophisticated network generation (e.g., hierarchical SBM, multiplex layers) and edge weight assignment based on the `pasted_content.txt` specifications.")
    print("3. **Parameter Calibration:** The `w_E, w_D, w_P` in `SEDPNRModel` and `m` in `build_fractal_network` need to be calibrated using real-world data and a Genetic Algorithm (GA) with Dynamic Time Warping (DTW) as described in the plan.")
    print("4. **Output Analysis:** Develop tools to analyze and visualize the simulation results (e.g., spread curves, state distributions over time).")
    print("5. **Policy Interventions:** Design mechanisms within the ABM to simulate policy interventions and evaluate their impact on disinformation spread.")


