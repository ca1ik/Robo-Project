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
num_episodes = 50
seed = 679
log_interval = 10
epsilon = 0.99
lr = 0.002

# Create environment
env = gym.make('MountainCar-v0')
state, _ = env.reset(seed=seed)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def epsilon_value(epsilon, decay=0.995):
    return max(0.1, epsilon * decay)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, state_inputs):
        x = F.relu(self.fc1(state_inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.critic_head(x), x

    def act(self, state_inputs, eps):
        value, x = self(state_inputs)
        x = F.softmax(self.actor_head(x), dim=-1)
        m = torch.distributions.Categorical(x)
        if random.random() > eps:
            action = m.sample()
        else:
            action = torch.randint(0, num_actions, (1,))
        return value, action, m.log_prob(action)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=lr)

def perform_updates():
    r = 0
    saved_actions = model.action_history
    returns = []
    rewards = model.rewards_achieved
    policy_losses = []
    critic_losses = []

    for i in rewards[::-1]:
        r = gamma * r + i
        returns.insert(0, r)
    returns = torch.tensor(returns)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # Calculate policy loss
        if log_prob.size() == torch.Size([1]):
            policy_losses.append(-log_prob * advantage)

        # Calculate value loss
        if value.size() == torch.Size([1]):
            critic_losses.append(F.mse_loss(value, torch.tensor([R])))

    if not policy_losses or not critic_losses:
        print("Warning: Empty policy_losses or critic_losses. Skipping update.")
        return 0.0

    try:
        policy_loss = torch.stack(policy_losses).sum()
        critic_loss = torch.stack(critic_losses).sum()
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
        return 0.0

    optimizer.zero_grad()
    loss = policy_loss + critic_loss
    loss.backward()
    optimizer.step()

    del model.rewards_achieved[:]
    del model.action_history[:]
    return loss.item()

def plot_rewards_with_label(episode_rewards, algorithm_name):
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label=algorithm_name)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc='upper right')
    plt.text(
        0.95, 0.95, algorithm_name,
        horizontalalignment='right', verticalalignment='top',
        transform=plt.gca().transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
    )
    plt.grid()
    plt.show()

def main():
    eps = epsilon
    plot_rewards = []

    for i_episode in range(num_episodes):
        state, _ = env.reset(seed=seed)
        ep_reward = 0
        done = False
        counter = 0

        while not done:
            state = torch.from_numpy(state).float()
            value, action, log_prob = model.act(state, eps)
            model.action_history.append(SavedAction(log_prob, value))

            state, reward, done, truncated, _ = env.step(action.item())
            reward += abs(state[1]) * 100  # Reward shaping for faster movement
            done = done or truncated

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1

            if counter % 5 == 0:
                perform_updates()

        plot_rewards.append(ep_reward)
        eps = epsilon_value(eps)

        print(f"Episode: {i_episode + 1}, Reward: {ep_reward:.2f}, Epsilon: {eps:.2f}")

    algorithm_name = "A2C - Mountain Car"
    plot_rewards_with_label(plot_rewards, algorithm_name)

if __name__ == '__main__':
    main()