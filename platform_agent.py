"""
This module defines the PlatformAgent class, which represents a social media
platform in the agent-based model.
"""
import numpy as np


class PlatformAgent:
    """
    A strategic agent representing the social media platform.
    Uses a Stackelberg game model to set its moderation policy to maximize utility.
    """

    def __init__(self, platform_name: str):
        self.name = platform_name
        # Utility parameters: cost of moderation vs. cost of reputational damage
        self.moderation_cost_factor = 1.5
        self.damage_cost_factor = 2.0
        self.engagement_factor = 1.0

        self.current_moderation_level = 0.1  # M in [0, 1]

    def _calculate_utility(self, moderation_level: float, engagement: float, spread_of_disinfo: float) -> float:
        """ Platform's utility function: U = Engagement - Cost(Moderation) - Cost(Damage) """
        engagement_utility = self.engagement_factor * engagement
        moderation_cost = self.moderation_cost_factor * (moderation_level ** 2)
        damage_cost = self.damage_cost_factor * spread_of_disinfo
        return engagement_utility - moderation_cost - damage_cost

    def optimize_policy(self, total_agents: int, num_sharing_disinfo: int, avg_engagement: float):
        """
        Finds the optimal moderation level M* by anticipating network reaction.
        This is a simplified Stackelberg optimization.
        """
        best_utility = -np.inf
        best_M = self.current_moderation_level

        # Test a range of possible moderation levels
        for M_test in np.linspace(0.0, 1.0, 11):
            # --- Anticipate reaction ---
            # Simple model: high moderation reduces sharing and engagement
            anticipated_spread = (num_sharing_disinfo /
                                  total_agents) * (1 - M_test)
            anticipated_engagement = avg_engagement * (1 - 0.5 * M_test)

            utility = self._calculate_utility(
                M_test, anticipated_engagement, anticipated_spread)

            if utility > best_utility:
                best_utility = utility
                best_M = M_test

        self.current_moderation_level = best_M
        print(
            f"Platform '{self.name}' updated moderation policy to: {self.current_moderation_level:.2f}")
