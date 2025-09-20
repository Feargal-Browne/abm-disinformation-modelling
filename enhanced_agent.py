import numpy as np
import torch
from typing import Dict

class EnhancedDisinformationAgent:
    """
    Advanced agent with:
    - Quantum-inspired belief state (GPU-accelerated)
    - Multiplex reputation system (platform-specific)
    - Evolutionary Game Theory strategies
    - T4/Colab optimized (PyTorch tensors, vectorized ops)
    """

    def __init__(self, agent_id: int, rng: np.random.Generator, platforms: list, pmt_params: dict, strategy: str, device="cuda"):
        self.id = agent_id
        self.rng = rng
        self.platforms = platforms
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Quantum-inspired belief state [|Skeptic>, |Believer>] as complex tensor
        self.belief_state = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.complex64, device=self.device)

        # Reputation per platform stored as GPU tensor
        self.reputation = {
            p: torch.tensor(rng.uniform(0.3, 0.7), dtype=torch.float32, device=self.device)
            for p in platforms
        }

        # Strategy/game theory attributes
        self.strategy = strategy
        self.payoff = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Core parameters
        self.pmt = pmt_params
        self.base_beta = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.gamma = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def measure_belief(self) -> str:
        """ Collapse quantum belief state into Skeptic or Believer. """
        prob_skeptic = torch.abs(self.belief_state[0])**2
        if torch.rand(1, device=self.device).item() < prob_skeptic.item():
            return "Skeptic"
        else:
            return "Believer"

    def update_belief(self, information_matrix: torch.Tensor):
        """
        Apply an information operator (rotation matrix) to update belief state.
        Fact-checks rotate towards Skeptic, endorsements towards Believer.
        """
        self.belief_state = information_matrix.to(self.device) @ self.belief_state
        self.belief_state /= torch.norm(self.belief_state)

    def calculate_share_utility(self, platform: str, moderation_level: float) -> torch.Tensor:
        """
        Expected utility of sharing, based on platform reputation and strategy.
        """
        rep = self.reputation[platform]
        potential_gain = rep * (1 - moderation_level)
        potential_loss = (1 - rep) * moderation_level

        if self.strategy == "AggressiveSharer":
            return 2.0 * potential_gain - 0.5 * potential_loss
        elif self.strategy == "CautiousVerifier":
            return 0.5 * potential_gain - 2.0 * potential_loss
        else:
            return potential_gain - potential_loss

    def decide_to_share(self, platform: str, moderation_level: float) -> bool:
        """ Decide whether to share based on utility > 0. """
        utility = self.calculate_share_utility(platform, moderation_level)
        return (utility > 0).item()

    def update_reputation(self, platform: str, outcome_is_successful: bool):
        """
        Update reputation using exponential smoothing:
        R(t+1) = λ * R(t) + (1-λ) * outcome
        """
        lambda_persistence = 0.9
        rep = self.reputation[platform]
        outcome_val = torch.tensor(1.0 if outcome_is_successful else 0.0, device=self.device)
        new_rep = lambda_persistence * rep + (1 - lambda_persistence) * outcome_val
        self.reputation[platform] = torch.clamp(new_rep, 0.0, 1.0)

# Example usage (Colab/T4 optimized)
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    agent = EnhancedDisinformationAgent(agent_id=1, rng=rng, platforms=["Twitter", "Facebook"], pmt_params={}, strategy="AggressiveSharer")

    # Example information operator (rotation matrix)
    info_matrix = torch.tensor([[0.9, -0.1],[0.1, 0.9]], dtype=torch.complex64)
    agent.update_belief(info_matrix)

    print("Belief measurement:", agent.measure_belief())
    print("Decision to share:", agent.decide_to_share("Twitter", moderation_level=0.2))
    agent.update_reputation("Twitter", outcome_is_successful=True)
    print("Updated reputation (Twitter):", agent.reputation["Twitter"].item())




