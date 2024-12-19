import argparse
import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt

# Argümanları tanımlama
parser = argparse.ArgumentParser(description='PyTorch Advantage Actor-Critic Example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='NU',
                    help='number of episodes (default: 1000)')
parser.add_argument('--seed', type=int, default=679, metavar='N',
                    help='random seed (default: 679)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# Çevreyi oluşturma ve seed ayarları
env = gym.make('MountainCar-v0')

# Seed ayarlarını yapma (sadece bir kez)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

num_inputs = env.observation_space.shape[0]  # Giriş boyutu (örneğin, 2)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.actor = nn.Linear(128, env.action_space.n)
        self.critic = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.xavier_uniform_(self.critic.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return value, action, m.log_prob(action), m.entropy()

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def perform_updates(saved_actions, rewards, gamma):
    '''
    ActorCritic ağının parametrelerini güncelleme
    '''
    R = 0
    returns = []
    policy_losses = []
    critic_losses = []
    entropies = []

    # Geriye doğru toplam ödülü hesapla
    for r in rewards[::-1]:
        R = gamma * R + r
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)

    # Normalize returns if std is not zero
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    for (log_prob, value, entropy), R in zip(saved_actions, returns):
        advantage = R - value.item()
        # Policy loss
        policy_losses.append(-log_prob * advantage)
        # Value loss
        critic_losses.append(F.mse_loss(value, torch.tensor([R])))
        # Entropy bonus
        entropies.append(-0.05 * entropy)  # Entropy bonus artırıldı

    # Toplam kaybı bulma
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum() + torch.stack(entropies).sum()
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def main():
    losses = []
    counters = []
    plot_rewards = []
    average_rewards = []

    for i_episode in range(1, args.num_episodes + 1):
        if i_episode % args.log_interval == 0:
            print(f"Starting Episode {i_episode}")

        counter = 0
        state, _ = env.reset()  # Seed'i burada geçirmiyoruz
        ep_reward = 0
        done = False

        # Aksiyon geçmişi ve ödülleri başlatma
        saved_actions = []
        rewards = []

        while not done:
            state_tensor = torch.from_numpy(state).float()
            value, action, log_prob, entropy = model.act(state_tensor)
            saved_actions.append((log_prob, value, entropy))
            # Ajan aksiyonu alır
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            ep_reward += reward
            counter += 1

        # Episode sonunda tüm aksiyonları güncelle
        loss = perform_updates(saved_actions, rewards, args.gamma)

        if i_episode % args.log_interval == 0:
            losses.append(loss)
            counters.append(counter)
            plot_rewards.append(ep_reward)
            avg_reward = np.mean(plot_rewards[-args.log_interval:])
            average_rewards.append(avg_reward)
            print(f"Episode {i_episode}\tLast loss: {loss:.4f}\tTotal Reward: {ep_reward}\tAverage Reward: {avg_reward:.2f}")

    # Kayıp grafiğini çizme
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.title('Loss over Episodes')
    plt.savefig('loss1.png')

    # Zaman adımlarını çizme
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Timesteps')
    plt.plot(counters)
    plt.title('Timesteps per Episode')
    plt.savefig('timestep.png')

    # Toplam ödülleri çizme
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(plot_rewards)
    plt.title('Total Rewards per Episode')
    plt.savefig('rewards.png')

    # Ortalama ödülleri çizme
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Average Rewards')
    plt.plot(average_rewards)
    plt.title('Average Rewards')
    plt.savefig('average_rewards.png')

    env.close()

if __name__ == '__main__':
    main()
