# Snake Q-Learning Agent

This project implements a reinforcement learning agent that learns to play Snake using Q-learning and Gymnasium.

## Hyperparameters

The agent's learning behavior is controlled by the following hyperparameters:

### Learning Parameters

**`learning_rate: float = 0.1`**
- Controls how much the agent updates its Q-values after each experience
- Range: 0.0 to 1.0
- Higher values (0.5-1.0): Agent learns faster but may be unstable
- Lower values (0.01-0.1): Agent learns slower but more consistently
- Default 0.1 provides balanced learning speed and stability

**`discount_factor: float = 0.95`**
- Determines how much the agent values future rewards vs immediate rewards
- Range: 0.0 to 1.0
- 0.0: Only cares about immediate rewards (myopic)
- 1.0: Future rewards are as important as immediate ones
- Default 0.95 means future rewards are valued at 95% of immediate rewards

### Exploration Parameters

**`epsilon: float = 1.0`**
- Starting exploration rate (probability of taking random actions)
- Range: 0.0 to 1.0
- 1.0: 100% random exploration at the beginning
- 0.0: 0% exploration (pure exploitation)
- Default 1.0 ensures thorough exploration during early training

**`epsilon_min: float = 0.01`**
- Minimum exploration rate that epsilon will decay to
- Range: 0.0 to 1.0
- Prevents epsilon from reaching zero, maintaining some exploration
- Default 0.01 keeps 1% exploration even after extensive training

**`epsilon_decay: float = 0.995`**
- Rate at which exploration decreases after each episode
- Range: 0.0 to 1.0
- Formula: epsilon = epsilon * epsilon_decay
- Default 0.995 takes ~1400 episodes to decay from 1.0 to 0.01

### Training Control

**`save_interval: int = 1000`**
- How frequently (in episodes) to save the model during training
- Prevents loss of training progress if training is interrupted
- Default 1000 saves every 1000 episodes

## Training Schedule

With default parameters, the agent follows this learning schedule:
- **Episodes 1-500**: Heavy exploration (epsilon > 0.6)
- **Episodes 500-1000**: Balanced exploration/exploitation (epsilon 0.6-0.3)
- **Episodes 1000-1500**: Light exploration (epsilon 0.3-0.1)
- **Episodes 1500+**: Minimal exploration (epsilon < 0.1)
