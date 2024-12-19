import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import time
from collections import deque
import random

# Neural network for the actor (policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log standard deviation heads
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

# Neural network for the critic (Q-value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (torch.FloatTensor(state), torch.FloatTensor(action), 
                torch.FloatTensor(reward), torch.FloatTensor(next_state),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

# SAC agent class
class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        
        # Copy critic parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter for exploration
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)
        
        # Update critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            q1_next, q2_next = self.critic_target(next_state_batch, next_action)
            q_next = torch.min(q1_next, q2_next)
            q_target = reward_batch.unsqueeze(1) + \
                      (1 - done_batch.unsqueeze(1)) * self.gamma * \
                      (q_next - self.alpha * next_log_prob)
        
        q1, q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_new, log_prob = self.actor.sample(state_batch)
        q1_new, q2_new = self.critic(state_batch, action_new)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), 
                                     self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + 
                                  (1 - self.tau) * target_param.data)

# Training loop
def train():
    env = gym.make('CarRacing-v3', continuous=True)
    state_dim = 96 * 96 * 3  # Flattened image dimensions
    action_dim = 3  # Steering, gas, brake
    
    agent = SACAgent(state_dim, action_dim)
    episodes = 100
    max_steps = 1000
    
    # Statistics tracking
    rewards_history = []
    success_count = 0
    total_reward = 0
    start_time = time.time()
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        state = state.reshape(-1)  # Flatten the image
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = next_state.reshape(-1)  # Flatten the image
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update statistics
        rewards_history.append(episode_reward)
        total_reward += episode_reward
        if episode_reward > 900:  # Consider episode successful if reward > 900
            success_count += 1
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = total_reward / (episode + 1)
            training_time = time.time() - start_time
            print(f"Episode: {episode + 1}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Success Rate: {success_count/(episode + 1):.2%}")
            print(f"Training Time: {training_time:.2f}s")
    
    return rewards_history, success_count, total_reward/episodes, training_time

# Run training
rewards_history, success_count, avg_reward, training_time = train()