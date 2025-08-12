import numpy as np
import random
from typing import Tuple, Dict, Any
from collections import defaultdict
import pickle
import os
from cycle_prevention import CyclePrevention


class QLearningAgent:
    def __init__(
        self,
        grid_size: int = 15,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(4))
        
        # Cycle prevention system
        self.cycle_prevention = CyclePrevention(
            max_history_length=8,
            cycle_penalty=-5
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.scores = []

    def get_state_key(self, observation: np.ndarray) -> str:
        """Convert observation to a hashable state key using relative positions."""
        # Find snake head position
        head_pos = None
        food_pos = None
        obstacles = []  # Fixed: collect ALL obstacles
        snake_body = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if observation[i, j] == 2:  # Head
                    head_pos = (i, j)
                elif observation[i, j] == 3:  # Food
                    food_pos = (i, j)
                elif observation[i, j] == 1:  # Body
                    snake_body.append((i, j))
                elif observation[i, j] == 4:  # Obstacles
                    obstacles.append((i, j))
        
        if head_pos is None or food_pos is None:
            return "invalid_state"
        
        # Calculate relative food position
        food_rel = (food_pos[0] - head_pos[0], food_pos[1] - head_pos[1])
        
        # Check for immediate dangers (walls, body, obstacles)
        dangers = self._get_immediate_dangers(head_pos, snake_body, obstacles)
        
        # Create state representation
        state_features = {
            'food_rel_row': np.clip(food_rel[0], -self.grid_size, self.grid_size),
            'food_rel_col': np.clip(food_rel[1], -self.grid_size, self.grid_size),
            'danger_up': dangers['up'],
            'danger_right': dangers['right'],
            'danger_down': dangers['down'],
            'danger_left': dangers['left'],
            'snake_length': len(snake_body) + 1
        }
        
        return str(sorted(state_features.items()))

    def _get_immediate_dangers(self, head_pos: Tuple[int, int], snake_body: list, obstacles: list) -> Dict[str, bool]:
        """Check for immediate dangers in each direction."""
        row, col = head_pos
        dangers = {}
        
        # Check each direction
        directions = {
            'up': (row - 1, col),
            'right': (row, col + 1),
            'down': (row + 1, col),
            'left': (row, col - 1)
        }
        
        for direction, (new_row, new_col) in directions.items():
            # Check wall collision
            wall_danger = (new_row < 0 or new_row >= self.grid_size or 
                          new_col < 0 or new_col >= self.grid_size)
            
            # Check body collision
            body_danger = (new_row, new_col) in snake_body

            # Check obstacle collision
            obstacle_danger = (new_row, new_col) in obstacles
            
            dangers[direction] = wall_danger or body_danger or obstacle_danger
        
        return dangers

    def choose_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy strategy with cycle prevention."""
        state_key = self.get_state_key(observation)
        head_pos = self._get_head_position(observation)
        
        if training and random.random() < self.epsilon:
            # Explore: choose random action but avoid immediate dangers and cycles
            return self._choose_safe_random_action(observation)
        else:
            # Exploit: choose best action with cycle prevention
            q_values = self.q_table[state_key].copy()
            
            # Apply cycle penalty using cycle prevention system
            q_values = self.cycle_prevention.apply_cycle_penalty(
                q_values, head_pos, self.grid_size
            )
            
            # Choose action with highest Q-value
            return np.argmax(q_values)

    def update_q_table(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        """Update Q-table using Q-learning update rule."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Next Q-value (0 if terminal state)
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update
        target_q = reward + self.discount_factor * next_max_q
        self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def record_episode(self, total_reward: float, episode_length: int, score: int):
        """Record episode statistics."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.scores.append(score)

    def get_stats(self, window: int = 100) -> Dict[str, float]:
        """Get training statistics over the last window episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_scores = self.scores[-window:]
        
        stats = {
            'avg_reward': np.mean(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'avg_score': np.mean(recent_scores),
            'max_score': max(recent_scores) if recent_scores else 0,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table)
        }
        
        # Add cycle prevention stats
        cycle_info = self.cycle_prevention.get_cycle_info()
        stats.update(cycle_info)
        
        return stats

    def save_model(self, filepath: str):
        """Save the Q-table and agent parameters."""
        model_data = {
            'q_table': dict(self.q_table),
            'grid_size': self.grid_size,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'scores': self.scores
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load the Q-table and agent parameters."""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(4), model_data['q_table'])
        self.grid_size = model_data['grid_size']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.epsilon_min = model_data['epsilon_min']
        self.epsilon_decay = model_data['epsilon_decay']
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_lengths = model_data.get('episode_lengths', [])
        self.scores = model_data.get('scores', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)} states")
        print(f"Current epsilon: {self.epsilon:.3f}")

    def reset_episode_history(self):
        """Reset position history for new episode."""
        self.cycle_prevention.reset_episode_history()
    
    def _get_head_position(self, observation: np.ndarray) -> Tuple[int, int]:
        """Extract head position from observation."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if observation[i, j] == 2:  # Head
                    return (i, j)
        return None
    
    
    def _choose_safe_random_action(self, observation: np.ndarray) -> int:
        """Choose random action that avoids immediate dangers and cycles."""
        head_pos = self._get_head_position(observation)
        
        # Get danger information from state
        dangers = self._extract_dangers_from_state(observation)
        
        # Get safe actions using cycle prevention system
        safe_actions = self.cycle_prevention.get_safe_actions(
            head_pos, dangers, self.grid_size
        )
        
        # Choose from safe actions, or any action if none are safe
        if safe_actions:
            return random.choice(safe_actions)
        else:
            return random.randint(0, 3)
    
    def _extract_dangers_from_state(self, observation: np.ndarray) -> Dict[str, bool]:
        """Extract danger information from observation."""
        head_pos = self._get_head_position(observation)
        if head_pos is None:
            return {'up': True, 'right': True, 'down': True, 'left': True}
        
        snake_body = []
        obstacles = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if observation[i, j] == 1:  # Body
                    snake_body.append((i, j))
                elif observation[i, j] == 4:  # Obstacles
                    obstacles.append((i, j))
        
        return self._get_immediate_dangers(head_pos, snake_body, obstacles)
    
    def update_position_history(self, head_pos: Tuple[int, int]):
        """Update position history for cycle detection."""
        self.cycle_prevention.update_position_history(head_pos)
    
    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.scores = []