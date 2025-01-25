import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
import time
from collections import deque
import random

# Neural network for the actor (policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # Define CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute the size of the flattened CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            cnn_output = self.conv(dummy_input)
            self.cnn_output_size = cnn_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)

        # To compute log_prob
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds with correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

# Neural network for the critic (Q-value)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # CNN for state
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Compute the size of the flattened CNN output
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim)
            cnn_output = self.conv(dummy_input)
            self.cnn_output_size = cnn_output.view(1, -1).size(1)

        # Q1 network
        self.q1_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 network
        self.q2_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        q1 = self.q1_fc(x)
        q2 = self.q2_fc(x)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        # Reshape to (C, H, W) and normalize
        state = torch.FloatTensor(np.array(state)).reshape(batch_size, 3, 96, 96) / 255.0
        next_state = torch.FloatTensor(np.array(next_state)).reshape(batch_size, 3, 96, 96) / 255.0
        return (state, 
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)),
                next_state,
                torch.FloatTensor(np.array(done)))

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

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


def train():
    env = gym.make('CarRacing-v3', continuous=True)

    state_dim = (3, 96, 96)  # (Channels, Height, Width)
    action_dim = 3  # Steering, gas, brake

    agent = SACAgent(state_dim, action_dim)
    episodes = 100
    max_steps = 1000

    rewards_history = [1, 2, 3, 4, 5]  # Example rewards history
    success_count = 3
    total_reward = sum(rewards_history)
    episodes = len(rewards_history)
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        state = state.transpose(2, 0, 1) / 255.0  # Transpose and normalize
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            action = np.clip(action, -1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.transpose(2, 0, 1) / 255.0  # Transpose and normalize

            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()

            state = next_state
            episode_reward += reward

            if episode <= 5:
                env.render()

            if done:
                break

        rewards_history.append(episode_reward)
        total_reward += episode_reward
        if episode_reward > 900:  # success criteria
            success_count += 1

        if episode % 10 == 0:
            avg_reward = total_reward / episode
            training_time = time.time() - start_time
            print(f"Episode: {episode}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Success Rate: {success_count / episode:.2%}")
            print(f"Training Time: {training_time:.2f}s")

    return rewards_history, success_count, total_reward / episodes, time.time() - start_time


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