import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Ortam ve parametreler
env = gym.make("MountainCarContinuous-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Hyperparameters
lr = 3e-4  # learning rate
gamma = 0.99  # discount factor
tau = 0.005  # target network update rate
alpha = 0.2  # Entropi katsayısı
hidden_dim = 256  # hidden layer dimension
replay_buffer_size = 100000  # replay buffer size
batch_size = 64

# Replay buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = []
        self.max_size = size
        self.ptr = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.ptr = (self.ptr + 1) % self.max_size
            self.buffer[self.ptr] = transition

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

# Ağ tanımları
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SACAgent:
    def __init__(self):
        self.actor = MLP(state_dim, action_dim, hidden_dim)
        self.critic_1 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.critic_2 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.target_critic_1 = MLP(state_dim + action_dim, 1, hidden_dim)
        self.target_critic_2 = MLP(state_dim + action_dim, 1, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Hedef ağları başlat
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        action = action.detach().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, 0.1, size=action_dim)
        return np.clip(action, -action_bound, action_bound)

    def train(self):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        # Replay buffer'dan örnek al
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Kritik ağların güncellenmesi
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q1 = self.target_critic_1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.target_critic_2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2) - alpha * next_actions
            q_target = rewards + gamma * (1 - dones) * next_q

        q1 = self.critic_1(torch.cat([states, actions], dim=1))
        q2 = self.critic_2(torch.cat([states, actions], dim=1))
        critic_1_loss = nn.MSELoss()(q1, q_target)
        critic_2_loss = nn.MSELoss()(q2, q_target)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Aktör ağının güncellenmesi
        actor_loss = -self.critic_1(torch.cat([states, self.actor(states)], dim=1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Kritik hedef ağların güncellenmesi
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Eğitim döngüsü
agent = SACAgent()
episodes = 50

rewards = []
for episode in range(episodes):
    state, _ = env.reset()  # Yeni Gym API'siyle uyumlu
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated  # Yeni Gym API için gerekli

        # Replay buffer'a geçişi ekleme
        agent.replay_buffer.add((state, action, reward, next_state, done))

        # Ağı güncelleme
        agent.train()

        # Durum güncelleme
        state = next_state
        episode_reward += reward

    rewards.append(episode_reward)
    print(f"Episode: {episode + 1}, Reward: {episode_reward}")

env.close()

# Eğitim sonuçlarını görselleştir
plt.figure(figsize=(12, 6))
plt.plot(rewards, label='SAC - Mountain Car')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('SAC Training on Mountain Car')
plt.grid()
plt.show()
