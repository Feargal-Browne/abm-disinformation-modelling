# Agent-Based Model for Disinformation Modelling

This repository contains a comprehensive Agent-Based Model (ABM) framework designed to simulate the propagation of disinformation. The framework is modular, allowing for the simulation of different scenarios and the evaluation of various policy interventions.

## Project Structure

-   `abm_framework.py`: A simple agent-based model.
-   `enhanced_abm_framework.py`: A more advanced agent-based model with features such as multiplex networks, evolutionary dynamics, and platform agents.
-   `sedpnr_model.py`: The core SEDPNR (Susceptible, Exposed, Doubtful, Positively Infected, Negatively Infected, Recovered) model.
-   `enhanced_agent.py`: Defines the `EnhancedDisinformationAgent` class, which represents a highly advanced agent in the agent-based model.
-   `platform_agent.py`: Defines the `PlatformAgent` class, which represents a social media platform in the agent-based model.
-   `evolutionary_dynamics.py`: Defines the `EvolutionaryDynamics` class, which manages the evolution of agent strategies based on payoffs.
-   `policy_interventions.py`: A script for simulating various policy interventions in the agent-based model.
-   `ipc_regression.py`: A script for predicting agent parameters (beta and gamma) from latent trait scores and network size.
-   `irt_analysis.py`: A conceptual framework for Item Response Theory (IRT) analysis.
-   `model_calibration.py`: A script for calibrating the agent-based model using a genetic algorithm.
-   `network_generator.py`: A script for generating various types of complex networks for the agent-based model.
-   `requirements.txt`: A list of all the Python dependencies required to run the code.

## Getting Started

### Local Environment

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/abm-disinformation-modelling.git
    cd abm-disinformation-modelling
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Google Colab

1.  **Open a new Colab notebook.**

2.  **Clone the repository:**
    ```python
    !git clone https://github.com/your-username/abm-disinformation-modelling.git
    %cd abm-disinformation-modelling
    ```

3.  **Install the dependencies:**
    ```python
    !pip install -r requirements.txt
    ```

## Running the Simulations

The repository contains two main simulation scripts: `abm_framework.py` and `enhanced_abm_framework.py`.

### Simple Simulation

To run the simple agent-based model, execute the following command in your terminal:

```bash
python abm_framework.py
```

### Advanced Simulation

To run the advanced agent-based model, execute the following command in your terminal:

```bash
python enhanced_abm_framework.py
```

### Other Scripts

The other scripts in the repository can be run individually to explore their functionality. For example, to run the model calibration script, execute the following command:

```bash
python model_calibration.py
```

**Note:** The scripts are designed to be run from the root of the repository.
