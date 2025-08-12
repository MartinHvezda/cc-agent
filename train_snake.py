import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import argparse
from snake_env import SnakeEnv
from q_learning_agent import QLearningAgent


def train_agent(
    episodes: int = 10000,
    grid_size: int = 15,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    save_interval: int = 1000,
    model_path: str = "models/snake_q_learning.pkl",
    load_model: bool = False
):
    """Train the Q-learning agent to play Snake."""
    
    # Initialize environment and agent
    env = SnakeEnv(grid_size=grid_size, render_mode=None)
    agent = QLearningAgent(
        grid_size=grid_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    # Load existing model if requested
    if load_model:
        agent.load_model(model_path)
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Discount factor: {discount_factor}")
    print(f"Initial epsilon: {epsilon}")
    print("-" * 50)
    
    for episode in range(episodes):
        # Reset environment
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.choose_action(observation, training=True)
            
            # Take step
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            agent.update_q_table(observation, action, reward, next_observation, done)
            
            # Update state
            observation = next_observation
            total_reward += reward
            steps += 1
        
        # Record episode stats
        agent.record_episode(total_reward, steps, info['score'])
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 100 == 0:
            stats = agent.get_stats(window=100)
            print(f"Episode {episode + 1:5d} | "
                  f"Avg Reward: {stats['avg_reward']:6.2f} | "
                  f"Avg Score: {stats['avg_score']:5.2f} | "
                  f"Max Score: {stats['max_score']:2.0f} | "
                  f"Epsilon: {stats['epsilon']:.3f} | "
                  f"Q-States: {stats['q_table_size']:5d}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model(model_path)
            plot_training_progress(agent, save_path=f"plots/training_progress_{episode + 1}.png")
    
    # Final save and evaluation
    agent.save_model(model_path)
    print(f"\nTraining completed! Model saved to {model_path}")
    
    # Plot final results
    plot_training_progress(agent, save_path="plots/final_training_progress.png")
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(env, agent, episodes=100)
    
    env.close()


def evaluate_agent(env: SnakeEnv, agent: QLearningAgent, episodes: int = 100):
    """Evaluate the trained agent."""
    scores = []
    episode_lengths = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Choose action without exploration
            action = agent.choose_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        scores.append(info['score'])
        episode_lengths.append(steps)
    
    print(f"Evaluation Results ({episodes} episodes):")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate (Score > 0): {np.mean([s > 0 for s in scores]) * 100:.1f}%")


def plot_training_progress(agent: QLearningAgent, save_path: str = "plots/training_progress.png"):
    """Plot training progress metrics."""
    if len(agent.episode_rewards) < 10:
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Snake Q-Learning Training Progress', fontsize=16)
    
    # Moving average window
    window = min(100, len(agent.episode_rewards) // 10)
    episodes = range(len(agent.episode_rewards))
    
    # Plot 1: Rewards
    axes[0, 0].plot(episodes, agent.episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, color='red', label=f'{window}-episode MA')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scores
    axes[0, 1].plot(episodes, agent.scores, alpha=0.3, color='green', label='Raw')
    if len(agent.scores) >= window:
        moving_avg = np.convolve(agent.scores, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], moving_avg, color='red', label=f'{window}-episode MA')
    axes[0, 1].set_title('Snake Scores (Food Eaten)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Episode Lengths
    axes[1, 0].plot(episodes, agent.episode_lengths, alpha=0.3, color='purple', label='Raw')
    if len(agent.episode_lengths) >= window:
        moving_avg = np.convolve(agent.episode_lengths, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(episodes[window-1:], moving_avg, color='red', label=f'{window}-episode MA')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Score Distribution (histogram)
    axes[1, 1].hist(agent.scores, bins=max(10, max(agent.scores) + 1), alpha=0.7, color='orange')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training progress plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Q-Learning agent for Snake game')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size for the game')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--discount-factor', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--save-interval', type=int, default=1000, help='Model save interval')
    parser.add_argument('--model-path', type=str, default='models/snake_q_learning.pkl', help='Model save path')
    parser.add_argument('--load-model', action='store_true', help='Load existing model')
    
    args = parser.parse_args()
    
    # Create directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    train_agent(
        episodes=args.episodes,
        grid_size=args.grid_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        save_interval=args.save_interval,
        model_path=args.model_path,
        load_model=args.load_model
    )


if __name__ == "__main__":
    main()