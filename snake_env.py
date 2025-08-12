import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Tuple, List, Optional


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size: int = 15, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid representation
        # 0=empty, 1=snake body, 2=snake head, 3=food, 4=obstacle
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32
        )
        
        # Game state
        self.snake_pos = []
        self.food_pos = None
        self.obstacles = []
        self.direction = 1  # Initial direction: right
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        
        # Pygame setup for rendering
        if self.render_mode == "human":
            pygame.init()
            self.cell_size = 30
            self.screen_size = self.grid_size * self.cell_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize snake in center
        center = self.grid_size // 2
        self.snake_pos = [(center, center)]
        self.direction = 1  # right
        self.score = 0
        self.steps = 0
        
        # Generate obstacles
        self._generate_obstacles()
        
        # Place food
        self._place_food()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: int):
        self.steps += 1
        
        # Update direction (prevent reversing)
        if self._is_valid_direction(action):
            self.direction = action
        
        # Move snake
        head = self.snake_pos[0]
        new_head = self._get_new_head(head, self.direction)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            terminated = True
            reward = -50
        # Check collision with self
        elif new_head in self.snake_pos:
            terminated = True
            reward = -20
        # Check collision with obstacles
        elif new_head in self.obstacles:
            terminated = True
            reward = -40
        else:
            terminated = False
            self.snake_pos.insert(0, new_head)
            
            # Check if food eaten
            if new_head == self.food_pos:
                self.score += 1
                reward = 20
                self._place_food()
            else:
                # Remove tail if no food eaten
                self.snake_pos.pop()
                reward = -0.1  # Small negative reward to encourage efficiency
        
        # Check if max steps reached
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            x, y = pos[1] * self.cell_size, pos[0] * self.cell_size
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Head brighter
            pygame.draw.rect(self.screen, color, (x, y, self.cell_size, self.cell_size))
        
        # Draw obstacles
        for pos in self.obstacles:
            x, y = pos[1] * self.cell_size, pos[0] * self.cell_size
            pygame.draw.rect(self.screen, (128, 128, 128), (x, y, self.cell_size, self.cell_size))
        
        # Draw food
        if self.food_pos:
            x, y = self.food_pos[1] * self.cell_size, self.food_pos[0] * self.cell_size
            pygame.draw.rect(self.screen, (255, 165, 0), (x, y, self.cell_size, self.cell_size))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_rgb_array(self):
        canvas = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Draw snake
        for i, pos in enumerate(self.snake_pos):
            if i == 0:  # Head
                canvas[pos[0], pos[1]] = [0, 255, 0]
            else:  # Body
                canvas[pos[0], pos[1]] = [0, 200, 0]
        
        # Draw obstacles
        for pos in self.obstacles:
            canvas[pos[0], pos[1]] = [128, 128, 128]
        
        # Draw food
        if self.food_pos:
            canvas[self.food_pos[0], self.food_pos[1]] = [255, 165, 0]
        
        return canvas

    def _get_observation(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Mark snake body
        for i, pos in enumerate(self.snake_pos):
            if i == 0:  # Head
                grid[pos[0], pos[1]] = 2
            else:  # Body
                grid[pos[0], pos[1]] = 1
        
        # Mark obstacles
        for pos in self.obstacles:
            grid[pos[0], pos[1]] = 4
        
        # Mark food
        if self.food_pos:
            grid[self.food_pos[0], self.food_pos[1]] = 3
        
        return grid

    def _place_food(self):
        empty_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.snake_pos and (i, j) not in self.obstacles:
                    empty_cells.append((i, j))
        
        if empty_cells:
            self.food_pos = random.choice(empty_cells)
    
    def _generate_obstacles(self):
        """Generate random obstacles at the start of each episode."""
        self.obstacles = []
        num_obstacles = max(1, self.grid_size // 5)  # Scale obstacles with grid size
        
        # Get all possible positions
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        
        # Remove snake starting position and adjacent cells
        center = self.grid_size // 2
        forbidden_positions = set()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                pos = (center + di, center + dj)
                if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                    forbidden_positions.add(pos)
        
        available_positions = [pos for pos in all_positions if pos not in forbidden_positions]
        
        # Randomly select obstacle positions
        if len(available_positions) >= num_obstacles:
            self.obstacles = random.sample(available_positions, num_obstacles)

    def _get_new_head(self, head: Tuple[int, int], direction: int) -> Tuple[int, int]:
        # 0=up, 1=right, 2=down, 3=left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        delta = directions[direction]
        return (head[0] + delta[0], head[1] + delta[1])

    def _is_valid_direction(self, new_direction: int) -> bool:
        # Prevent reversing into self (opposite directions)
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}
        return new_direction != opposite.get(self.direction)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake_pos)
        }

    def close(self):
        if hasattr(self, "screen"):
            pygame.display.quit()
            pygame.quit()