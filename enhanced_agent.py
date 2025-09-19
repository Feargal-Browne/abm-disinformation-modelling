# enhanced_agent.py
import numpy as np
from typing import Dict

class EnhancedDisinformationAgent:
    """
    A highly advanced agent with a complex cognitive and strategic architecture.
    - Belief is modeled as a quantum superposition state.
    - Reputation and strategy are platform-specific (Multiplex).
    - Makes strategic decisions based on utility calculation.
    - Possesses a strategy attribute for Evolutionary Game Theory.
    """
    def __init__(self, agent_id: int, rng: np.random.Generator, platforms: list, pmt_params: dict, strategy: str):
        self.id = agent_id
        self.rng = rng
        self.platforms = platforms
        
        # --- Quantum Cognition: Belief State ---
        # State vector [alpha, beta] for |Skeptic> and |Believer>
        # Initialized to a neutral superposition state
        self.belief_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)

        # --- Multiplex Game Theory: Platform-specific attributes ---
        self.reputation: Dict[str, float] = {p: rng.uniform(0.3, 0.7) for p in platforms}
        
        # --- Evolutionary Game Theory (EGT) ---
        self.strategy = strategy # e.g., 'AggressiveSharer', 'CautiousVerifier'
        self.payoff = 0.0 # Payoff accumulated for strategy success

        # --- Core Agent Parameters (from survey/IPC) ---
        self.pmt = pmt_params # Contains severity, vulnerability, etc.
        self.base_beta = 0.0 # Propensity to share, determined by strategy
        self.gamma = 0.0 # Propensity to recover
        
    def measure_belief(self) -> str:
        """ Collapses the quantum belief state into a classical outcome. """
        prob_skeptic = np.abs(self.belief_state[0])**2
        if self.rng.random() < prob_skeptic:
            return 'Skeptic'
        else:
            return 'Believer'

    def update_belief(self, information_matrix: np.ndarray):
        """
        Updates the belief state by applying an information operator (rotation matrix).
        Fact-checks rotate towards |Skeptic>, endorsements rotate towards |Believer>.
        """
        self.belief_state = information_matrix @ self.belief_state
        # Normalize the state vector to maintain unit length
        self.belief_state /= np.linalg.norm(self.belief_state)

    def calculate_share_utility(self, platform: str, moderation_level: float) -> float:
        """
        Calculates the expected utility of sharing a piece of content.
        This is a core game-theoretic calculation.
        """
        # Simplified utility: potential reputation gain vs. potential loss
        # Higher own reputation -> higher potential gain
        # Higher moderation -> higher potential loss if debunked
        potential_gain = self.reputation[platform] * (1 - moderation_level)
        potential_loss = (1 - self.reputation[platform]) * moderation_level
        
        # Strategy modifies the calculation
        if self.strategy == 'AggressiveSharer':
            # Overweighs potential gain
            return 2.0 * potential_gain - 0.5 * potential_loss
        elif self.strategy == 'CautiousVerifier':
            # Overweighs potential loss
            return 0.5 * potential_gain - 2.0 * potential_loss
        else:
            return potential_gain - potential_loss

    def decide_to_share(self, platform: str, moderation_level: float) -> bool:
        """ Makes the strategic decision to share or not. """
        utility = self.calculate_share_utility(platform, moderation_level)
        # Share if utility is positive
        return utility > 0

    def update_reputation(self, platform: str, outcome_is_successful: bool):
        """
        Updates reputation based on the outcome of a sharing event.
        Uses the formula: R(t+1) = lambda*R(t) + (1-lambda)*outcome
        """
        lambda_persistence = 0.9 # How much old reputation persists
        current_rep = self.reputation[platform]
        outcome_val = 1.0 if outcome_is_successful else 0.0
        
        new_rep = lambda_persistence * current_rep + (1 - lambda_persistence) * outcome_val
        self.reputation[platform] = np.clip(new_rep, 0.0, 1.0)


