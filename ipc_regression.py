import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- IMPORTANT NOTE ON EXECUTION ENVIRONMENT ---
# This script outlines the Individual Parameter Contribution (IPC) Regression.
# The actual execution will depend on the availability of real survey data
# and the latent trait scores derived from the IRT analysis.
# For computationally intensive tasks, Google Colab is recommended.
# ------------------------------------------------

class IPCRegression:
    def __init__(self):
        self.model_beta = None
        self.model_gamma = None

    def generate_dummy_data(self, num_participants=1000):
        """
        Generates dummy data for latent trait scores, network size, and
        conceptual beta/gamma values for demonstration.
        In a real scenario, these would come from IRT analysis and behavioral proxies.
        """
        print(f"Generating dummy data for {num_participants} participants for IPC regression...")

        # Dummy latent trait scores for Big 5 (5 traits)
        # Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
        latent_traits = np.random.randn(num_participants, 5)
        df_latent_traits = pd.DataFrame(latent_traits, columns=[
            'Openness_Score', 'Conscientiousness_Score', 'Extraversion_Score',
            'Agreeableness_Score', 'Neuroticism_Score'
        ])

        # Dummy network size (e.g., log-normal distribution for scale-free like properties)
        network_size = np.exp(np.random.randn(num_participants) * 0.5 + 2) # Mean around e^2 ~ 7.4
        df_network_size = pd.DataFrame(network_size, columns=['Network_Size'])

        # Dummy behavioral proxies for beta and gamma
        # These would be derived from survey questions related to spread/disengagement tendencies
        # For demonstration, let's make them somewhat dependent on traits and network size
        # Example: higher openness/extraversion -> higher beta
        # Example: higher conscientiousness -> higher gamma
        dummy_beta_proxy = (1 + df_latent_traits['Openness_Score'] * 0.1 + 
                            df_latent_traits['Extraversion_Score'] * 0.15 + 
                            np.log(df_network_size['Network_Size']) * 0.2 + 
                            np.random.randn(num_participants) * 0.1)
        dummy_gamma_proxy = (1 + df_latent_traits['Conscientiousness_Score'] * 0.1 + 
                             df_latent_traits['Neuroticism_Score'] * 0.05 + 
                             np.random.randn(num_participants) * 0.05)

        # Ensure proxies are positive for log-link regression
        dummy_beta_proxy = np.maximum(0.01, dummy_beta_proxy)
        dummy_gamma_proxy = np.maximum(0.01, dummy_gamma_proxy)

        df_proxies = pd.DataFrame({
            'Beta_Proxy': dummy_beta_proxy,
            'Gamma_Proxy': dummy_gamma_proxy
        })

        return pd.concat([df_latent_traits, df_network_size, df_proxies], axis=1)

    def fit_ipc_models(self, data):
        """
        Fits regression models to predict beta and gamma parameters
        from latent trait scores and network size.
        Uses a log-link function as specified in the planning document.
        """
        print("Fitting IPC regression models...")

        X = data[[
            'Openness_Score', 'Conscientiousness_Score', 'Extraversion_Score',
            'Agreeableness_Score', 'Neuroticism_Score', 'Network_Size'
        ]]
        y_beta = np.log(data['Beta_Proxy']) # Apply log-link
        y_gamma = np.log(data['Gamma_Proxy']) # Apply log-link

        # Fit Linear Regression for Beta
        self.model_beta = LinearRegression()
        self.model_beta.fit(X, y_beta)
        print("Beta model fitted.")
        print(f"Beta Model Coefficients: {self.model_beta.coef_}")
        print(f"Beta Model Intercept: {self.model_beta.intercept_}")

        # Fit Linear Regression for Gamma
        self.model_gamma = LinearRegression()
        self.model_gamma.fit(X, y_gamma)
        print("Gamma model fitted.")
        print(f"Gamma Model Coefficients: {self.model_gamma.coef_}")
        print(f"Gamma Model Intercept: {self.model_gamma.intercept_}")

    def predict_agent_parameters(self, agent_data):
        """
        Predicts beta and gamma for new agents based on their latent traits and network size.
        `agent_data` should be a DataFrame with columns matching X used in fitting.
        """
        if self.model_beta is None or self.model_gamma is None:
            raise ValueError("IPC models have not been fitted. Run `fit_ipc_models` first.")

        print("Predicting agent-specific beta and gamma parameters...")
        predicted_log_beta = self.model_beta.predict(agent_data)
        predicted_log_gamma = self.model_gamma.predict(agent_data)

        # Convert back from log-link using exponential
        predicted_beta = np.exp(predicted_log_beta)
        predicted_gamma = np.exp(predicted_log_gamma)

        return predicted_beta, predicted_gamma


if __name__ == "__main__":
    ipc_regressor = IPCRegression()

    # Step 1: Generate dummy data (replace with real data from IRT and survey)
    dummy_ipc_data = ipc_regressor.generate_dummy_data(num_participants=1000)
    print("Dummy IPC data head:\n", dummy_ipc_data.head())

    # Step 2: Fit the IPC regression models
    ipc_regressor.fit_ipc_models(dummy_ipc_data)

    # Step 3: Demonstrate prediction for a few hypothetical agents
    print("\nDemonstrating prediction for hypothetical agents...")
    # Create some hypothetical agent data (e.g., for 3 agents)
    hypothetical_agent_data = pd.DataFrame({
        'Openness_Score': [1.5, -0.5, 0.8],
        'Conscientiousness_Score': [0.2, 1.0, -0.7],
        'Extraversion_Score': [0.9, -1.2, 0.1],
        'Agreeableness_Score': [-0.3, 0.7, 0.5],
        'Neuroticism_Score': [0.6, -0.8, 1.1],
        'Network_Size': [50, 500, 100]
    })

    predicted_beta_agents, predicted_gamma_agents = ipc_regressor.predict_agent_parameters(hypothetical_agent_data)

    print("Predicted Beta for hypothetical agents:", predicted_beta_agents)
    print("Predicted Gamma for hypothetical agents:", predicted_gamma_agents)

    # --- Outline for integrating with the full pipeline ---
    print("\n--- Outline for Integrating IPC Regression into the Full Pipeline ---")
    print("1. After collecting survey data, perform IRT analysis (using `irt_analysis.py` or similar) to obtain:")
    print("   - Latent trait scores for Big 5 personality traits for each participant.")
    print("   - Latent trait scores for Schwartz Value Circumplex for each participant.")
    print("2. Extract network size data and behavioral proxies (e.g., spread tendency, disengagement tendency) from the survey.")
    print("3. Combine these into a single DataFrame for the `IPCRegression` class.")
    print("4. Call `fit_ipc_models` to train the regression models.")
    print("5. During Agent-Based Model (ABM) initialization (Phase 6), for each of the 200,000 simulated agents:")
    print("   - Stochastically assign latent trait scores and network size based on the distributions from the survey data.")
    print("   - Use the trained `ipc_regressor.predict_agent_parameters` method to calculate the unique `beta_i` and `gamma_i` for each simulated agent.")
    print("   - These `beta_i` and `gamma_i` values, along with other agent-specific parameters, will then be fed into the `SEDPNRModel`.")


