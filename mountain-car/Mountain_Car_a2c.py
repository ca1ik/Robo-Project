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

# Seed ayarlarını yapma
# Gym v0.26 ve üzeri için env.reset()'te seed kullanılır
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

num_inputs = env.observation_space.shape[0]  # Giriş boyutu (örneğin, 2)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def epsilon_value(eps):
    return 0.99 * eps

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fc3.weight)
        num_actions = env.action_space.n

        self.actor_head = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.critic_head.weight)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, state_inputs):
        x = F.relu(self.fc1(state_inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.critic_head(x), x

    def act(self, state_inputs):
        value, x = self(state_inputs)
        probs = F.softmax(self.actor_head(x), dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return value, action, m.log_prob(action)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.002)

def perform_updates():
    '''
    ActorCritic ağının parametrelerini güncelleme
    '''
    R = 0
    saved_actions = model.action_history
    returns = []
    rewards = model.rewards_achieved
    policy_losses = []
    critic_losses = []

    # Geriye doğru toplam ödülü hesapla
    for r in rewards[::-1]:
        R = args.gamma * R + r
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if returns.std() != 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize returns

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        # Policy loss
        policy_losses.append(-log_prob * advantage)
        # Value loss
        critic_losses.append(F.mse_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    # Toplam kaybı bulma
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optimizer.step()
    # Aksiyon geçmişi ve ödülleri temizle
    del model.rewards_achieved[:]
    del model.action_history[:]
    return loss.item()

def main():
    losses = []
    counters = []
    plot_rewards = []

    for i_episode in range(1, args.num_episodes + 1):
        if i_episode % args.log_interval == 0:
            print(f"Starting Episode {i_episode}")

        counter = 0
        # Çevreyi resetlerken seed kullanmayın, sadece başlangıçta ayarlayın
        state, _ = env.reset(seed=args.seed)  # Gym v0.26 ve üzeri
        ep_reward = 0
        done = False

        while not done:
            state_tensor = torch.from_numpy(state).float()
            value, action, ac_log_prob = model.act(state_tensor)
            model.action_history.append(SavedAction(ac_log_prob, value))
            # Ajan aksiyonu alır
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1

            if counter % 5 == 0:
                ''' Her 5 zaman adımında bir geri yayılım yaparak
                    yüksek korelasyonlu durumları önler '''
                loss = perform_updates()

        # Episode sonunda kalan aksiyonları güncelle
        if len(model.rewards_achieved) > 0:
            loss = perform_updates()

        if i_episode % args.log_interval == 0:
            losses.append(loss)
            counters.append(counter)
            plot_rewards.append(ep_reward)
            print(f"Episode {i_episode}\tLast loss: {loss:.4f}\tTotal Reward: {ep_reward}")

    # Kayıp grafiğini çizme
    plt.figure()
    plt.xlabel('Episodes (x{})'.format(args.log_interval))
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('loss1.png')

    # Zaman adımlarını çizme
    plt.figure()
    plt.xlabel('Episodes (x{})'.format(args.log_interval))
    plt.ylabel('Timesteps')
    plt.plot(counters)
    plt.savefig('timestep.png')

    # Toplam ödülleri çizme
    plt.figure()
    plt.xlabel('Episodes (x{})'.format(args.log_interval))
    plt.ylabel('Rewards')
    plt.plot(plot_rewards)
    plt.savefig('rewards.png')

    env.close()

if __name__ == '__main__':
    main()
