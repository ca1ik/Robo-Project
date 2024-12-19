import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Advantage Actor Critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--num_episodes', type=int, default=50, metavar='N',
                        help='number of episodes (default: 50)')
    parser.add_argument('--seed', type=int, default=679, metavar='S',
                        help='random seed (default: 679)')
    return parser.parse_args()

# Environment and hyperparameters
args = parse_args()
env = gym.make('MountainCar-v0')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def epsilon_decay(epsilon, decay_rate=0.99):
    return epsilon * decay_rate

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.critic_head(x), x

    def select_action(self, state, epsilon):
        value, x = self(state)
        probs = F.softmax(self.actor_head(x), dim=-1)
        m = Categorical(probs)

        if random.random() > epsilon:
            action = m.sample()
        else:
            action = torch.randint(0, num_actions, (1,))

        return value, action, m.log_prob(action)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def perform_updates():
    R = 0
    gamma = args.gamma
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    for (log_prob, value), R in zip(model.saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.mse_loss(value, torch.tensor([R])))

    if policy_losses and value_losses:
        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

def main():
    epsilon = 0.99
    episode_rewards = []

    for episode in range(args.num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        ep_reward = 0
        done = False

        while not done:
            value, action, log_prob = model.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated

            model.saved_actions.append(SavedAction(log_prob, value))
            model.rewards.append(reward)

            state = torch.FloatTensor(next_state)
            ep_reward += reward

        episode_rewards.append(ep_reward)
        perform_updates()
        epsilon = epsilon_decay(epsilon)

        print(f"Episode {episode + 1}: Reward = {ep_reward:.2f}, Epsilon = {epsilon:.2f}")

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('A2C Training on MountainCar-v0')
    plt.show()

if __name__ == '__main__':
    main()
