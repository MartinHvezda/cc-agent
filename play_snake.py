import gymnasium as gym
import pygame
import time
import argparse
from snake_env import SnakeEnv
from q_learning_agent import QLearningAgent

def play_agent(
    model_path: str = "models/snake_q_learning.pkl",
    grid_size: int = 10,
    fps: int = 10,
    episodes: int = 5
):
    """Watch trained agent play Snake."""
    env = SnakeEnv(grid_size=grid_size, render_mode="human")
    agent = QLearningAgent(grid_size=grid_size)
    
    # Load trained model
    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"Model file {model_path} not found! Please train the agent first.")
        return
    
    print("AI Agent Playing")
    print(f"Loaded model: {model_path}")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print("Press ESC to quit, SPACE to reset game")
    print("-" * 40)
    
    episode = 0
    observation, info = env.reset()
    running = True
    clock = pygame.time.Clock()
    paused = False
    
    while running and episode < episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Reset game
                    observation, info = env.reset()
                    episode += 1
                    print(f"Starting episode {episode + 1}")
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
        
        if not paused:
            # Agent chooses action
            action = agent.choose_action(observation, training=False)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"Episode {episode + 1} finished! Score: {info['score']}, Steps: {info['steps']}")
                episode += 1
                if episode < episodes:
                    time.sleep(2)  # Pause before next episode
                    observation, info = env.reset()
                    print(f"Starting episode {episode + 1}")
        
        clock.tick(fps)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Play Snake game')
    parser.add_argument('--model-path', type=str, default='models/snake_q_learning.pkl',
                        help='Path to trained model')
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to play')
    
    args = parser.parse_args()

    play_agent(
        model_path=args.model_path,
        grid_size=args.grid_size,
        fps=args.fps,
        episodes=args.episodes
    )

if __name__ == "__main__":
    main()