from collections import defaultdict
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any

class GridWorldAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent for GridWorld with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The GridWorld environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def _get_state_key(self, obs: Dict[str, np.ndarray]) -> Tuple[int, int, int, int]:
        """Convert the dictionary observation into a tuple that can be used as a dict key"""
        agent_pos = obs["agent"]
        target_pos = obs["target"]
        return (int(agent_pos[0]), int(agent_pos[1]), 
                int(target_pos[0]), int(target_pos[1]))

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        
        Args:
            obs: Dictionary containing agent and target positions
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # with probability (1 - epsilon) act greedily (exploit)
        state_key = self._get_state_key(obs)
        return int(np.argmax(self.q_values[state_key]))

    def update(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Dict[str, np.ndarray],
    ):
        """Updates the Q-value of an action."""
        state_key = self._get_state_key(obs)
        next_state_key = self._get_state_key(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_state_key])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state_key][action]
        )

        self.q_values[state_key][action] = (
            self.q_values[state_key][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Decay epsilon value"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
