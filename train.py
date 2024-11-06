import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import grid_gym
from agent import GridWorldAgent

# Create environment
env = gym.make('grid_gym/GridWorld-v0', size=5)

# Initialize agent
agent = GridWorldAgent(
    env=env,
    learning_rate=0.1,
    initial_epsilon=1.0,
    epsilon_decay=0.001,
    final_epsilon=0.1,
    discount_factor=0.95
)

# Training parameters
n_episodes = 1000
max_steps = 100
all_rewards = []

# Training loop
for episode in range(n_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not (terminated or truncated) and steps < max_steps:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        agent.update(obs, action, reward, terminated, next_obs)
        obs = next_obs
        total_reward += reward
        steps += 1
    
    agent.decay_epsilon()
    all_rewards.append(total_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(all_rewards[-100:])
        print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

env.close()
