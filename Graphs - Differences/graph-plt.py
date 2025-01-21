import gymnasium as gym #problematic lit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time    
import plt    #pip
import gym



def plot_rewards_with_label(episode_rewards, algorithm_name):
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label=algorithm_name)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()


    if __name__ == "__main__":
     rewards_history, success_count, avg_reward, training_time = train()
    algorithm_name = 'SAC | Car Racing'
    plot_rewards_with_label(rewards_history, algorithm_name)