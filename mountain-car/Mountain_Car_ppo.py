import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Ortak Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
gamma = 0.99
tau = 0.005  # target network update rate for DDPG and SAC

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (torch.tensor(states, dtype=torch.float32).to(device),
                torch.tensor(actions, dtype=torch.float32).to(device),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(next_states, dtype=torch.float32).to(device),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device))

# Ortak Ağlar
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))
        return self.max_action * torch.tanh(self.fc3(action))

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def update(self, replay_buffer, batch_size, old_log_probs):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Log-probabiliteleri hesapla
        log_probs = self.actor(states)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Politika kaybı (Clipped Objective)
        advantage = rewards - self.critic(states, actions)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value kaybı
        value_loss = nn.functional.mse_loss(self.critic(states, actions), rewards)

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

def train(agent, env, replay_buffer, agent_name, num_episodes=50, batch_size=64, reward_threshold=-0.01):
    episode_rewards = []
    old_log_probs = torch.zeros(batch_size, 1).to(device)  # PPO için başlangıç değeri

    for episode in range(num_episodes):
        state, _ = env.reset()  # state ve info'yu ayırıyoruz
        state = np.array(state, dtype=np.float32)  # numpy array'e dönüştürme
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)  # numpy array'e dönüştürme
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if len(replay_buffer.buffer) > batch_size:
                if agent_name == "PPO":
                    agent.update(replay_buffer, batch_size, old_log_probs)  # PPO için old_log_probs geçiriyoruz
                else:
                    agent.update(replay_buffer, batch_size)
            
            # Eğer episode tamamlanmışsa ya da zaman aşımına uğramışsa, döngüyü bitir
            if done or truncated:
                break
                
        episode_rewards.append(episode_reward)
        print(f"Agent: {agent_name} | Episode: {episode + 1} | Reward: {episode_reward}")
        
        # Eğer ödül eşiğine ulaşılmışsa eğitimi sonlandır
        if episode_reward >= reward_threshold:
            print(f"{agent_name} hedef ödül eşiğine ulaştı. Eğitim sonlandırılıyor.")
            break

    return episode_rewards


# Eğitim Başlatma
replay_buffer = ReplayBuffer()
ppo_agent = PPOAgent(state_dim, action_dim, max_action)

ppo_rewards = train(ppo_agent, env, replay_buffer, "PPO")

# Sonuçları Görselleştir
plt.plot(ppo_rewards, label='PPO')
plt.legend()
plt.show()