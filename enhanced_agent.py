import numpy as np

class DisinformationAgent:
    """
    Agent for ABM disinformation propagation with:
    - Diffusion of Innovations adopter categories (affects belief threshold)
    - Protection Motivation Theory (PMT) attributes and resistance calculation
    - Long-memory exposure effect (exposure_memory influences decay)
    - Non-linear social reinforcement via q-exponent
    """
    def __init__(self, agent_id: int, rng: np.random.Generator, 
                 adopter_dist=None,
                 initial_belief: float = 0.0,
                 ideology: float | None = None,
                 pmt_from_latent: dict | None = None):
        self.id = agent_id
        self.rng = rng
        self.belief = float(initial_belief)
        # Optional trait used for homophily rewiring (e.g., ideology [-3,3])
        self.ideology = float(ideology if ideology is not None else rng.normal(0, 1))

        # DOI: adopter category and belief threshold mapping
        self.adopter_category = self.assign_adopter_category(adopter_dist)
        self.belief_threshold = self.set_belief_threshold()

        # PMT: initialize from latent traits if provided, otherwise random in [0,1]
        if pmt_from_latent is None:
            self.perceived_severity = rng.random()
            self.perceived_vulnerability = rng.random()
            self.self_efficacy = rng.random()
            self.response_efficacy = rng.random()
        else:
            # Expect standardized inputs in [0,1]; clip just in case
            self.perceived_severity = float(np.clip(pmt_from_latent.get('severity', rng.random()), 0, 1))
            self.perceived_vulnerability = float(np.clip(pmt_from_latent.get('vulnerability', rng.random()), 0, 1))
            self.self_efficacy = float(np.clip(pmt_from_latent.get('self_efficacy', rng.random()), 0, 1))
            self.response_efficacy = float(np.clip(pmt_from_latent.get('response_efficacy', rng.random()), 0, 1))

        # Long memory counter of exposures to current topic
        self.exposure_memory = 0

        # Convenience cached status string for simple analytics
        self.status = self.update_status_from_belief()

    def assign_adopter_category(self, adopter_dist=None) -> str:
        if adopter_dist is None:
            adopter_dist = {
                'innovator': 0.025,
                'early_adopter': 0.135,
                'early_majority': 0.34,
                'late_majority': 0.34,
                'laggard': 0.16
            }
        cats = list(adopter_dist.keys())
        probs = np.array([adopter_dist[c] for c in cats], dtype=float)
        probs = probs / probs.sum()
        return self.rng.choice(cats, p=probs)

    def set_belief_threshold(self) -> float:
        thresholds = {
            'innovator': 0.20,
            'early_adopter': 0.40,
            'early_majority': 0.60,
            'late_majority': 0.80,
            'laggard': 0.95
        }
        return float(thresholds[self.adopter_category])

    def calculate_resistance(self) -> float:
        # PMT resistance: high threat appraisal and high coping appraisal -> higher resistance
        threat_appraisal = self.perceived_severity * self.perceived_vulnerability
        coping_appraisal = self.self_efficacy * self.response_efficacy
        resistance = float(np.clip(threat_appraisal * coping_appraisal, 0.0, 1.0))
        return resistance

    def update_status_from_belief(self) -> str:
        # Simplified mapping: belief above threshold => "Infected"; otherwise "Susceptible"
        return "Infected" if self.belief >= self.belief_threshold else "Susceptible"

    def decay_belief(self) -> None:
        # Long-memory influenced decay per pasted_content_2
        # Decay slower as exposure_memory grows
        decay_factor = 0.99 + (0.009 * (1 - (1 / (self.exposure_memory + 1))))
        decay_factor = float(np.clip(decay_factor, 0.0, 0.9999))
        if self.status != "Infected":  # don't decay strongly when firmly infected
            self.belief *= decay_factor
            self.belief = float(np.clip(self.belief, 0.0, 1.0))

    def apply_media_event(self, effect_strength: float, cognitive_ability: float | None = None):
        # Example: agents with higher cognitive ability respond more to fact-checks
        if cognitive_ability is None:
            cognitive_ability = float(self.rng.uniform(0, 1))
        if cognitive_ability > 0.7:
            self.belief *= (1.0 - float(np.clip(effect_strength, 0.0, 1.0)))
            self.belief = float(np.clip(self.belief, 0.0, 1.0))
            self.status = self.update_status_from_belief()

    def interact_update(self, infected_neighbor_beliefs: list[float],
                        base_persuasiveness: float,
                        q_exponent: float,
                        dynamic_msg_factor: float) -> None:
        """
        Update belief from neighbors using q-exponent social pressure and PMT resistance.
        infected_neighbor_beliefs: beliefs of neighbors currently "Infected"
        base_persuasiveness: baseline persuasiveness scalar in [0,1]
        q_exponent: q > 1 super-linear, q < 1 sub-linear
        dynamic_msg_factor: [0,1] scaler derived from IMT2-like message properties
        """
        num_infected = len(infected_neighbor_beliefs)
        if num_infected <= 0:
            return
        # Non-linear social pressure
        social_pressure = (num_infected ** q_exponent) * base_persuasiveness * dynamic_msg_factor
        social_pressure = float(np.clip(social_pressure, 0.0, 1.0))
        # Average source belief vs current belief
        source_belief = float(np.mean(infected_neighbor_beliefs))
        resistance = self.calculate_resistance()
        belief_change = (source_belief - self.belief) * social_pressure * (1.0 - resistance)
        # Apply update and memory increment
        self.belief = float(np.clip(self.belief + belief_change, 0.0, 1.0))
        if belief_change > 0:
            self.exposure_memory += 1
        self.status = self.update_status_from_belief()


