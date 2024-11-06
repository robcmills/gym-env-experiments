import matplotlib.pyplot as plt
import numpy as np

def plot(steps_per_episode, optimal_action_ratios):
    # Create the visualization
    plt.figure(figsize=(12, 5))

    # Plot 1: Steps per Episode
    plt.subplot(1, 2, 1)
    plt.plot(steps_per_episode, color='blue', alpha=0.6)
    plt.plot(np.convolve(steps_per_episode, np.ones(50)/50, mode='valid'), 
             color='red', linewidth=2, label='50-episode moving average')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Optimal Action Ratio
    plt.subplot(1, 2, 2)
    plt.plot(optimal_action_ratios, color='green', alpha=0.6)
    plt.plot(np.convolve(optimal_action_ratios, np.ones(50)/50, mode='valid'),
             color='red', linewidth=2, label='50-episode moving average')
    plt.xlabel('Episode')
    plt.ylabel('Optimal Action Ratio')
    plt.title('Ratio of Actions Moving Toward Target')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots.png', dpi=300, bbox_inches='tight')
    plt.close()
