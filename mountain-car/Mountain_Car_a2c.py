import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # BURADA EKLENDİ
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple

# Diğer kod aynı...


# Hyperparameters
gamma = 0.99
num_episodes = 50
seed = 679
log_interval = 10
epsilon = 0.99
lr_actor = 3e-4
lr_critic = 3e-4
tau = 0.005

# Create environment
env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Set seeds
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Replay buffer for sampling experiences
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.fc(state)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

# SAC Agent
class SACAgent:
    def __init__(self):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer()

        # Copy weights to target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(action_dim)
        return np.clip(action, -action_bound, action_bound)

    def update(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            q_target_1 = self.target_critic_1(next_states, next_actions)
            q_target_2 = self.target_critic_2(next_states, next_actions)
            q_target = rewards + gamma * (1 - dones) * torch.min(q_target_1, q_target_2)

        # Update critics
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss_1 = F.mse_loss(q1, q_target)
        critic_loss_2 = F.mse_loss(q2, q_target)

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Update actor
        actor_loss = -self.critic_1(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = SACAgent()
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, done))
        agent.update()
        state = next_state
        episode_reward += reward

    rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Plot Results
plt.plot(rewards, label="SAC - MountainCarContinuous")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Training Rewards")
plt.legend()
plt.show()
