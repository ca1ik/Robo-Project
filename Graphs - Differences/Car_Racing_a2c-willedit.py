import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

class A2CNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute feature vector size
        with torch.no_grad():
            feature_size = self.conv_layers(torch.zeros(1, *input_shape)).flatten().shape[0]
        
        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, num_actions),
            nn.Tanh()  # Use Tanh for continuous action space
        )
        
        # Critic head (value)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        features = self.conv_layers(x)
        shared_features = self.fc_shared(features)
        policy = self.actor(shared_features)
        value = self.critic(shared_features)
        return policy, value

class A2CAgent:
    def __init__(self, input_shape, action_space, learning_rate=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.num_actions = action_space.shape[0]
        
        self.network = A2CNetwork(input_shape, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state)
            policy = policy.cpu().numpy()[0]
        
        # Scale actions to match environment's action space
        action = np.clip(policy, 
                         self.action_space.low, 
                         self.action_space.high)
        return action
    
    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return returns
    
    def train(self, states, actions, rewards, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # Forward pass
        policies, values = self.network(states)
        values = values.squeeze()
        
        # Compute returns
        with torch.no_grad():
            _, last_value = self.network(states[-1].unsqueeze(0))
            last_value = last_value.item()
            returns = self.compute_returns(rewards, dones, last_value)
        
        # Compute advantages
        advantages = torch.FloatTensor(returns).to(self.device) - values
        
        # Actor loss (policy gradient)
        # Use Mean Squared Error for continuous action space
        actor_loss = F.mse_loss(policies, actions)
        
        # Critic loss (value regression)
        critic_loss = F.mse_loss(values, torch.FloatTensor(returns).to(self.device))
        
        # Total loss with entropy regularization
        loss = actor_loss + 0.5 * critic_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        return loss.item()

def preprocess_state(state):
    # Normalize state
    state = state.transpose(2, 0, 1)  # CHW format
    state = state / 255.0  # Normalize to [0, 1]
    return state

def train_a2c(env_name, num_episodes=100, max_steps=1000):
    env = gym.make(env_name, continuous=True)
    input_shape = (3, 96, 96)  # CarRacing-v3 observation shape
    
    agent = A2CAgent(input_shape, env.action_space)
    
    episode_rewards = []
    training_start_time = time.time()
    successful_episodes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        done = False
        truncated = False
        
        episode_states, episode_actions, episode_rewards_list, episode_dones = [], [], [], []
        
        for step in range(max_steps):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Preprocess next state
            next_state = preprocess_state(next_state)
            
            # Store experience
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards_list.append(reward)
            episode_dones.append(float(terminated or truncated))
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Train periodically or at episode end
            if len(episode_states) >= 5 or terminated or truncated:
                agent.train(episode_states, episode_actions, 
                            episode_rewards_list, episode_dones)
                episode_states, episode_actions, episode_rewards_list, episode_dones = [], [], [], []
            
            # Check for episode termination
            if terminated or truncated:
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        
        if episode_reward > 900:  # Consider episode successful if high reward
            successful_episodes += 1
        
        # Print progress periodically
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
    
    # Calculate training statistics
    training_time = time.time() - training_start_time

    # Example usage
    algorithm_name = 'A2C | Car Racing'
    plot_rewards_with_label(episode_rewards, algorithm_name)

def plot_rewards_with_label(episode_rewards, algorithm_name):
    
    # Plot rewards
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

    
    return agent, episode_rewards

# Run the training
if __name__ == "__main__":
    agent, rewards = train_a2c('CarRacing-v3', num_episodes=100)