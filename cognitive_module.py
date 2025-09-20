# cognitive_module.py
import numpy as np
import cupy as cp
from numba import jit

class CognitiveModule:
    """
    Manages an agent's cognitive state and decision-making processes.
    This module replaces the quantum cognition model with a more grounded,
    cognitively plausible architecture.
    """
    def __init__(self, agent_id: int, rng: np.random.Generator, pmt_params: dict, strategy: str):
        self.id = agent_id
        self.rng = rng

        # Belief is a continuous variable from -1 (disbelief) to 1 (belief)
        self.belief = self.rng.uniform(-0.1, 0.1)

        # Cognitive biases
        self.confirmation_bias_strength = self.rng.uniform(0.1, 0.5)
        self.ingroup_bias_strength = self.rng.uniform(0.1, 0.5)

        # Protection Motivation Theory (PMT) parameters
        self.pmt = pmt_params

        # Evolutionary Game Theory (EGT) strategy
        self.strategy = strategy
        self.payoff = 0.0

    def update_belief(self, message_belief_strength: float, message_emotion: float, source_is_ingroup: bool):
        """
        Updates the agent's belief based on an incoming message.
        """
        # Confirmation bias
        belief_congruence = 1 - abs(self.belief - message_belief_strength)
        confirmation_effect = self.confirmation_bias_strength * belief_congruence

        # Ingroup bias
        ingroup_effect = self.ingroup_bias_strength if source_is_ingroup else 0

        # Emotional arousal
        emotion_effect = message_emotion * 0.1

        # Combine effects and update belief
        belief_update = message_belief_strength + confirmation_effect + ingroup_effect + emotion_effect
        self.belief += belief_update
        self.belief = float(np.clip(self.belief, -1.0, 1.0))

    def decide_to_share(self, platform_moderation: float) -> bool:
        """
        Makes the decision to share based on belief, PMT, and strategy.
        """
        # High belief is a strong motivator to share
        if self.belief > 0.7:
            return True

        # PMT-based decision for less certain agents
        threat_appraisal = self.pmt['severity'] * self.pmt['vulnerability']
        coping_appraisal = self.pmt['self_efficacy'] * self.pmt['response_efficacy']

        if coping_appraisal > threat_appraisal:
            # If coping is high, less likely to share disinformation
            return False

        # Strategy-based decision
        if self.strategy == 'AggressiveSharer':
            return self.rng.random() < 0.8
        elif self.strategy == 'CautiousVerifier':
            return self.rng.random() < 0.2
        else:
            return False
