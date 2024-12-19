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

def normalize_state(state):
    position, velocity = state
    # MountainCar-v0 position range: [-1.2, 0.6], velocity range: [-0.07, 0.07]
    pos_min, pos_max = -1.2, 0.6
    vel_min, vel_max = -0.07, 0.07
    norm_pos = (position - pos_min) / (pos_max - pos_min)
    norm_vel = (velocity - vel_min) / (vel_max - vel_min)
    return np.array([norm_pos, norm_vel], dtype=np.float32)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.actor = nn.Linear(64, env.action_space.n)
        self.critic = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.actor.weight)
        nn.init.xavier_uniform_(self.critic.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return value, action, m.log_prob(action), m.entropy()

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Öğrenme oranını artırdık

def perform_updates(saved_actions, rewards, gamma):
    '''
    ActorCritic ağının parametrelerini güncelleme
    '''
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float)

    # Stack saved actions
    log_probs = torch.stack([action.log_prob for action in saved_actions])
    values = torch.stack([action.value.squeeze() for action in saved_actions])
    entropies = torch.stack([action.entropy for action in saved_actions])

    advantages = returns - values.detach()

    # Normalize advantages
    if advantages.std() > 0:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # Policy loss
    policy_losses = (-log_probs * advantages).sum()

    # Value loss
    value_losses = F.mse_loss(values, returns)

    # Entropy loss
    entropy_loss = -0.05 * entropies.sum()  # Entropy bonus artırıldı

    # Total loss
    loss = policy_losses + value_losses + entropy_loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Calculate individual losses for logging
    policy_loss_val = policy_losses.item()
    value_loss_val = value_losses.item()
    entropy_loss_val = entropy_loss.item()

    return loss.item(), policy_loss_val, value_loss_val, entropy_loss_val

def main():
    losses = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    counters = []
    plot_rewards = []
    average_rewards = []

    for i_episode in range(1, args.num_episodes + 1):
        if i_episode % args.log_interval == 0:
            print(f"Starting Episode {i_episode}")

        counter = 0
        state, _ = env.reset()  # Seed'i burada geçirmiyoruz
        state = normalize_state(state)
        ep_reward = 0
        done = False

        # Aksiyon geçmişi ve ödülleri başlatma
        saved_actions = []
        rewards = []

        while not done:
            state_tensor = torch.from_numpy(state).float()
            value, action, log_prob, entropy = model.act(state_tensor)
            saved_actions.append(SavedAction(log_prob, value, entropy))
            # Ajan aksiyonu alır
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            state = normalize_state(state)
            rewards.append(reward)
            ep_reward += reward
            counter += 1

        # Episode sonunda tüm aksiyonları güncelle
        loss, policy_loss_val, value_loss_val, entropy_loss_val = perform_updates(saved_actions, rewards, args.gamma)

        if i_episode % args.log_interval == 0:
            losses.append(loss)
            policy_losses.append(policy_loss_val)
            value_losses.append(value_loss_val)
            entropy_losses.append(entropy_loss_val)
            counters.append(counter)
            plot_rewards.append(ep_reward)
            avg_reward = np.mean(plot_rewards[-args.log_interval:])
            average_rewards.append(avg_reward)
            print(f"Episode {i_episode}\tLoss: {loss:.4f}\tPolicy Loss: {policy_loss_val:.4f}\tValue Loss: {value_loss_val:.4f}\tEntropy Loss: {entropy_loss_val:.4f}\tTotal Reward: {ep_reward}\tAverage Reward: {avg_reward:.2f}")

    # Kayıp grafiğini çizme
    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(losses, label='Total Loss')
    plt.plot(policy_losses, label='Policy Loss')
    plt.plot(value_losses, label='Value Loss')
    plt.plot(entropy_losses, label='Entropy Loss')
    plt.legend()
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
