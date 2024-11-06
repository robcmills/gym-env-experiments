import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import grid_gym
from agent import GridWorldAgent
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

env_name = "grid_gym/GridWorld-v0"
save_dir = "./saved_models"

def train_and_save_agent(total_timesteps=5000):
    """
    Train a PPO agent and save it to disk.
    
    Args:
        total_timesteps (int): Number of timesteps to train for
    
    Returns:
        model: Trained PPO model
    """
    # Create and wrap the environment
    env = gym.make(env_name, size=5)
    env = Monitor(env)
    
    # Initialize the agent
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(save_dir, f"{env_name}_ppo")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def load_and_evaluate_agent():
    """
    Load a saved agent and evaluate its performance.
    
    Returns:
        float: Mean reward
        float: Standard deviation of reward
    """
    # Create environment
    env = gym.make(env_name)
    
    # Load the saved model
    model_path = os.path.join(save_dir, f"{env_name}_ppo")
    model = PPO.load(model_path)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def test_saved_agent(episodes=3):
    """
    Load a saved agent and run it in the environment for visualization.
    
    Args:
        model_path (str): Path to the saved model
        episodes (int): Number of episodes to run
    """
    # Create environment
    env = gym.make(env_name, render_mode="rgb_array", size=5)
    env = RecordVideo(env, video_folder="renders", episode_trigger=lambda x: True)
    print(f"Environment created: {env_name}")
    
    # Load the saved model
    model_path = os.path.join(save_dir, f"{env_name}_ppo")
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    print("Model loaded")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        total_steps = 0
        max_steps = 20
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_scalar = action.item()
            obs, reward, terminated, truncated, _ = env.step(action_scalar)
            done = terminated or truncated or total_steps >= max_steps
            total_reward += reward
            total_steps += 1
            
        print(f"Episode {episode + 1} reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    train_and_save_agent()
    # mean_reward, std_reward = load_and_evaluate_agent()
    test_saved_agent()
