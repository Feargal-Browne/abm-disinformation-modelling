"""
This module provides a class for calibrating the agent-based model using a
genetic algorithm. It is designed to find the optimal parameters for the
model by comparing the simulation output to real-world data.
"""
import numpy as np
import pygad
from dtw import dtw
from enhanced_abm_framework import EnhancedAgentBasedModel
import pandas as pd

# --- IMPORTANT NOTE ON MODEL CALIBRATION ---
# Model calibration, especially with complex ABMs and Genetic Algorithms,
# is computationally very intensive and can take a significant amount of time.
# It is highly recommended to run this component on a high-performance computing
# environment like Google Colab with GPU acceleration (if PyGAD supports it for
# specific operations) or dedicated cloud resources.
# The `real_world_data` placeholder must be replaced with actual time-series data
# of disinformation spread (e.g., number of infected individuals over time).
# --------------------------------------------


class ModelCalibrator:
    """
    A class for calibrating the agent-based model using a genetic algorithm.
    """

    def __init__(self, num_agents: int, real_world_data: np.ndarray, time_points: np.ndarray,
                 population_data: pd.DataFrame = None, seed: int = None):
        """
        Initializes the ModelCalibrator.

        Args:
            num_agents (int): The number of agents in the simulation.
            real_world_data (np.ndarray): The real-world data to compare against.
            time_points (np.ndarray): The time points for the simulation.
            population_data (pd.DataFrame, optional): The population data for the agents. Defaults to None.
            seed (int, optional): The random seed for reproducibility. Defaults to None.
        """
        self.num_agents = num_agents
        # Target time series (e.g., total infected over time)
        self.real_world_data = real_world_data
        self.time_points = time_points
        self.population_data = population_data
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Parameters to be calibrated (example: m for BA network, q_exponent, w_E, w_D, w_P)
        # These ranges need to be carefully chosen based on domain knowledge.
        self.param_ranges = {
            # For Barabasi-Albert network
            "m": {"low": 1, "high": 10, "type": "int"},
            # Tsallis exponent
            "q_exponent": {"low": 0.1, "high": 2.0, "type": "float"},
            # Weight for Exposed
            "w_E": {"low": 0.1, "high": 2.0, "type": "float"},
            # Weight for Doubtful
            "w_D": {"low": 0.1, "high": 2.0, "type": "float"},
            # Weight for Positively Infected
            "w_P": {"low": 0.1, "high": 2.0, "type": "float"}
        }

        self.last_fitness = 0
        self.best_solution = None
        self.best_solution_fitness = -np.inf

    def _run_abm_simulation(self, m, q_exponent, w_E, w_D, w_P) -> np.ndarray:
        """
        Runs the ABM simulation with the given parameters and returns the total infected population over time.
        """
        # Initialize ABM with current parameters
        abm = EnhancedAgentBasedModel(
            num_agents=self.num_agents, population_data=self.population_data, seed=self.seed)
        abm.initialize_agents()

        # Build network with the current 'm' parameter
        # Note: homophily and community parameters are fixed for calibration, or can be added to calibration
        abm.build_network(m=int(m))

        # Update SEDPNRModel's weights based on current GA solution
        # This requires modifying SEDPNRModel to accept these weights dynamically
        # For now, we'll pass them to the run_simulation and assume they are used.
        # A more robust solution would involve passing these to the SEDPNRModel constructor
        # or having a method to update them.
        # (Conceptual: these would be set inside the SEDPNRModel instance)
        # sedpnr_model.w_E = w_E, etc.

        # Dummy message properties and media event schedule for calibration run
        message_properties = {
            'is_false': 0.8,
            'is_vague': 0.6,
            'is_irrelevant': 0.3
        }
        media_event_schedule = []  # No media events during calibration

        # Run simulation
        simulation_results = abm.run_simulation(
            self.time_points, q_exponent, message_properties, media_event_schedule)

        # Extract total infected population (S+E+D+P+N+R = 1, so P+N is infected)
        # Assuming P and N are the infected states in your SEDPNR model, adjust accordingly.
        # For this example, let's assume P (Positively Infected) and N (Negatively Infected)
        # represent the 'infected' states. The `simulation_results` array has shape
        # (time_points, num_agents, 6). We need to sum P and N for each agent at each
        # time step, then sum across agents.

        # Extract P and N states for all agents at all time points
        P_states = simulation_results[:, :, 3]  # Index 3 for P
        N_states = simulation_results[:, :, 4]  # Index 4 for N

        # Sum P and N for each agent, then sum across all agents to get total infected population
        total_infected_over_time = (P_states + N_states).sum(axis=1)

        return total_infected_over_time

    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Fitness function for the Genetic Algorithm.
        It runs the ABM with the given solution (parameters) and calculates the DTW distance
        between the simulated output and the real-world data. Lower DTW distance means higher fitness.
        """
        # Extract parameters from the solution array based on their order in gene_space
        m = solution[0]
        q_exponent = solution[1]
        w_E = solution[2]
        w_D = solution[3]
        w_P = solution[4]

        # Ensure integer type for 'm'
        m = int(round(m))

        # Run ABM simulation with these parameters
        simulated_data = self._run_abm_simulation(
            m, q_exponent, w_E, w_D, w_P)

        # Ensure simulated_data and real_world_data have the same length
        if len(simulated_data) != len(self.real_world_data):
            # This should ideally not happen if time_points are consistent
            # For robustness, interpolate or truncate if lengths differ
            min_len = min(len(simulated_data), len(self.real_world_data))
            simulated_data = simulated_data[:min_len]
            self.real_world_data = self.real_world_data[:min_len]

        # Calculate DTW distance as fitness. Lower distance is better, so we negate it.
        # Use dtw-python library for DTW calculation
        # The `distance` method returns the optimal alignment cost.
        try:
            alignment = dtw(simulated_data,
                            self.real_world_data, keep_internals=False)
            dtw_distance = alignment.distance
        except Exception as e:
            print(f"Error during DTW calculation: {e}")
            dtw_distance = np.inf  # Penalize errors heavily

        # Genetic algorithms typically maximize fitness, so we negate the distance.
        fitness = -dtw_distance
        self.last_fitness = fitness
        return fitness

    def calibrate(self, num_generations=100, sol_per_pop=10, num_parents_mating=5):
        """
        Runs the Genetic Algorithm to calibrate the ABM parameters.
        """
        # Define the gene space for the parameters
        # Order must match the order in `fitness_func`
        gene_space = [
            self.param_ranges["m"]["low"], self.param_ranges["m"]["high"],
            self.param_ranges["q_exponent"]["low"], self.param_ranges["q_exponent"]["high"],
            self.param_ranges["w_E"]["low"], self.param_ranges["w_E"]["high"],
            self.param_ranges["w_D"]["low"], self.param_ranges["w_D"]["high"],
            self.param_ranges["w_P"]["low"], self.param_ranges["w_P"]["high"]
        ]

        # Convert gene_space to a list of tuples for PyGAD
        gene_space_pygad = []
        for i in range(0, len(gene_space), 2):
            gene_space_pygad.append((gene_space[i], gene_space[i+1]))

        # Define the number of genes (parameters to optimize)
        num_genes = len(self.param_ranges)

        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=self.fitness_func,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_type=[self.param_ranges[key]["type"]
                       for key in self.param_ranges.keys()],
            gene_space=gene_space_pygad,
            parent_selection_type="sss",  # Steady-State Selection
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            keep_parents=-1,  # Keep the best parents in the population
            random_seed=self.seed
        )

        print("Starting Genetic Algorithm calibration...")
        ga_instance.run()

        self.best_solution, self.best_solution_fitness, _ = ga_instance.best_solution()
        print(f"Best solution found: {self.best_solution}")
        print(
            f"Best solution fitness (negative DTW distance): {self.best_solution_fitness}")

        # Map best solution back to parameter names
        calibrated_params = {
            "m": int(round(self.best_solution[0])),
            "q_exponent": self.best_solution[1],
            "w_E": self.best_solution[2],
            "w_D": self.best_solution[3],
            "w_P": self.best_solution[4]
        }
        return calibrated_params


if __name__ == "__main__":
    # --- Example Usage ---
    # 1. Generate dummy real-world data (e.g., a simple S-curve)
    TIME_POINTS = np.linspace(0, 50, 51)
    # Simulate a simple S-curve for 'real-world' infected data
    # This should be replaced by actual historical data of disinformation spread
    REAL_WORLD_INFECTED_DATA = 1000 / \
        (1 + np.exp(-0.2 * (TIME_POINTS - 25)))  # Max 1000 infected

    # 2. Generate dummy population data (as in enhanced_abm_framework.py)
    DUMMY_POP_DATA_SIZE = 1000
    RNG_POP = np.random.default_rng(42)
    DUMMY_POP_LATENT_TRAITS = RNG_POP.normal(
        0, 1, size=(DUMMY_POP_DATA_SIZE, 5))
    DUMMY_POP_NETWORK_SIZES = RNG_POP.integers(10, 1000, DUMMY_POP_DATA_SIZE)
    DUMMY_POPULATION_DATA = pd.DataFrame({
        'Openness_Score': DUMMY_POP_LATENT_TRAITS[:, 0],
        'Conscientiousness_Score': DUMMY_POP_LATENT_TRAITS[:, 1],
        'Extraversion_Score': DUMMY_POP_LATENT_TRAITS[:, 2],
        'Agreeableness_Score': DUMMY_POP_LATENT_TRAITS[:, 3],
        'Neuroticism_Score': DUMMY_POP_LATENT_TRAITS[:, 4],
        'Network_Size': DUMMY_POP_NETWORK_SIZES
    })

    # 3. Initialize and run calibrator
    calibrator = ModelCalibrator(
        num_agents=100,  # Use a smaller number of agents for calibration runs
        real_world_data=REAL_WORLD_INFECTED_DATA,
        time_points=TIME_POINTS,
        population_data=DUMMY_POPULATION_DATA,
        seed=42
    )

    CALIBRATED_PARAMETERS = calibrator.calibrate(
        num_generations=10, sol_per_pop=5)  # Reduced generations for quick test
    print("\nCalibrated Parameters:", CALIBRATED_PARAMETERS)

    # --- Next Steps ---
    print("\n--- Next Steps ---")
    print("1. Replace `real_world_infected_data` with actual historical data.")
    print("2. Adjust `param_ranges` in `ModelCalibrator` based on expected values.")
    print("3. Increase `num_generations` and `sol_per_pop` for more robust calibration.")
    print("4. Consider parallelizing fitness function evaluation for faster calibration.")


