import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple

# Hyperparameters
gamma = 0.99
num_episodes = 100
seed = 679
lr = 0.002
log_interval = 10
epsilon = 0.99

env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Seeding
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def epsilon_decay(epsilon, decay=0.995, min_epsilon=0.1):
    return max(min_epsilon, epsilon * decay)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.actor_head(x), self.critic_head(x)

    def act(self, state, epsilon):
        actor_logits, value = self.forward(state)
        probs = F.softmax(actor_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if random.random() > epsilon:
            action = dist.sample()
        else:
            action = torch.randint(0, num_actions, (1,))
        return action, dist.log_prob(action), value

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=lr)

def perform_updates(model, optimizer, saved_actions, rewards, gamma):
    R = 0
    returns = []
    policy_losses = []
    value_losses = []

    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.mse_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

def train():
    epsilon = 0.99
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        rewards = []
        saved_actions = []
        ep_reward = 0

        while True:
            action, log_prob, value = model.act(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)
            saved_actions.append(SavedAction(log_prob, value))

            state = torch.tensor(next_state, dtype=torch.float32)
            ep_reward += reward

            if done or truncated:
                perform_updates(model, optimizer, saved_actions, rewards, gamma)
                episode_rewards.append(ep_reward)
                epsilon = epsilon_decay(epsilon)
                print(f"Episode {episode + 1}: Reward = {ep_reward:.2f}, Epsilon = {epsilon:.2f}")
                break

    # Plot rewards
    plt.plot(episode_rewards, label="A2C - Mountain Car")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
