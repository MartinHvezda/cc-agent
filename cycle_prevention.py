import numpy as np
import random
from typing import Tuple, List, Dict
from collections import deque


class CyclePrevention:
    """Handles cycle detection and prevention for Snake RL agent."""
    
    def __init__(self, max_history_length: int = 8, cycle_penalty: float = -5):
        """
        Initialize cycle prevention system.
        
        Args:
            max_history_length: Maximum number of positions to remember
            cycle_penalty: Penalty applied to Q-values for cycle-creating actions
        """
        self.max_history_length = max_history_length
        self.cycle_penalty = cycle_penalty
        self.position_history = deque(maxlen=max_history_length)
    
    def reset_episode_history(self):
        """Reset position history for new episode."""
        self.position_history.clear()
    
    def update_position_history(self, head_pos: Tuple[int, int]):
        """
        Update position history for cycle detection.
        
        Args:
            head_pos: Current head position (row, col)
        """
        self.position_history.append(head_pos)
    
    def would_create_cycle(self, next_pos: Tuple[int, int], min_cycle_length: int = 4) -> bool:
        """
        Check if moving to next_pos would create a cycle.
        
        Args:
            next_pos: Position to check (row, col)
            min_cycle_length: Minimum positions needed to detect a cycle
            
        Returns:
            True if next_pos would create a cycle
        """
        if len(self.position_history) < min_cycle_length:
            return False
        
        # Check if next position was recently visited
        recent_positions = list(self.position_history)[-min_cycle_length:]
        return next_pos in recent_positions
    
    def apply_cycle_penalty(self, q_values: np.ndarray, head_pos: Tuple[int, int], 
                           grid_size: int) -> np.ndarray:
        """
        Apply cycle penalty to Q-values for actions that would create cycles.
        
        Args:
            q_values: Original Q-values for actions [up, right, down, left]
            head_pos: Current head position (row, col)
            grid_size: Size of the game grid
            
        Returns:
            Modified Q-values with cycle penalties applied
        """
        modified_q_values = q_values.copy()
        
        # Check each action
        for action in range(4):
            next_pos = self._get_next_position(head_pos, action, grid_size)
            if self.would_create_cycle(next_pos):
                modified_q_values[action] += self.cycle_penalty
        
        return modified_q_values
    
    def get_safe_actions(self, head_pos: Tuple[int, int], dangers: Dict[str, bool], 
                        grid_size: int) -> List[int]:
        """
        Get list of actions that are safe (no immediate danger and no cycle).
        
        Args:
            head_pos: Current head position (row, col)
            dangers: Dictionary of dangers by direction {'up': bool, ...}
            grid_size: Size of the game grid
            
        Returns:
            List of safe action indices
        """
        safe_actions = []
        direction_names = ['up', 'right', 'down', 'left']
        
        for action in range(4):
            direction = direction_names[action]
            next_pos = self._get_next_position(head_pos, action, grid_size)
            
            # Check if action is safe (no danger and no cycle)
            if not dangers[direction] and not self.would_create_cycle(next_pos):
                safe_actions.append(action)
        
        return safe_actions
    
    def _get_next_position(self, head_pos: Tuple[int, int], action: int, 
                          grid_size: int) -> Tuple[int, int]:
        """
        Calculate next position given current position and action.
        
        Args:
            head_pos: Current head position (row, col)
            action: Action to take (0=up, 1=right, 2=down, 3=left)
            grid_size: Size of the game grid
            
        Returns:
            Next position (row, col)
        """
        row, col = head_pos
        # 0=up, 1=right, 2=down, 3=left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        delta = directions[action]
        return (row + delta[0], col + delta[1])
    
    def get_cycle_info(self) -> Dict:
        """
        Get information about current cycle detection state.
        
        Returns:
            Dictionary with cycle detection statistics
        """
        return {
            'history_length': len(self.position_history),
            'max_history_length': self.max_history_length,
            'cycle_penalty': self.cycle_penalty,
            'recent_positions': list(self.position_history)[-4:] if self.position_history else []
        }