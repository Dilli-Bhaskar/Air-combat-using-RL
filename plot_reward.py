import numpy as np
import matplotlib.pyplot as plt

# Load rewards from .npy file
rewards = np.load('rewards_100.npy')
rewards_8 = np.load('rewards_8.npy')

# Define the window size for rolling mean and std
window_size = 100

# Calculate rolling mean and rolling standard deviation
rolling_mean = np.convolve(rewards, np.ones(window_size), 'valid') / window_size
reward_std = np.std([rewards[i:i+window_size] for i in range(len(rewards)-window_size)], axis=1)

rolling_mean_8 = np.convolve(rewards_8, np.ones(window_size), 'valid') / window_size
reward_std_8 = np.std([rewards_8[i:i+window_size] for i in range(len(rewards_8)-window_size)], axis=1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(rewards, color='#ADD8E6', alpha=0.6)
plt.plot(rewards_8, color='#FFB6C1', alpha=0.6)

plt.plot(np.arange(window_size-1, len(rewards)), rolling_mean, label='Ter_reward: 100', color='blue')
plt.plot(np.arange(window_size-1, len(rewards_8)), rolling_mean_8, label='Ter_reward: 8', color='red')

# Set titles and labels with increased font size
plt.title('Reward vs Episode', fontsize=18)
plt.xlabel('Episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)

# Increase font size of the legend
plt.legend(fontsize=12)

# Increase grid size and apply other settings
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
