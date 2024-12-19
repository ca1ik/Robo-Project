import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Neural Network for Actor-Critic architecture
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dims=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Fully connected layers
        with torch.no_grad():
            conv_output_size = self.conv_layers(torch.zeros(1, *input_dims)).shape[1]
        
        # Actor (Policy) network
        self.actor_fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU()
        )
        
        # Steer, Accelerate, Brake branches
        self.mu_steer = nn.Linear(hidden_dims, 1)  # Mean for steering
        self.sigma_steer = nn.Linear(hidden_dims, 1)  # Standard deviation for steering
        
        self.mu_throttle = nn.Linear(hidden_dims, 1)  # Mean for throttle
        self.sigma_throttle = nn.Linear(hidden_dims, 1)  # Standard deviation for throttle
        
        self.mu_brake = nn.Linear(hidden_dims, 1)  # Mean for brake
        self.sigma_brake = nn.Linear(hidden_dims, 1)  # Standard deviation for brake
        
        # Critic network (Value estimation)
        self.critic_fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)  # Value estimation
        )
        
    def forward(self, state):
        # Convolutional feature extraction
        conv_out = self.conv_layers(state)
        
        # Actor branches
        actor_features = self.actor_fc(conv_out)
        
        # Steer action distribution
        mu_steer = torch.tanh(self.mu_steer(actor_features))
        sigma_steer = F.softplus(self.sigma_steer(actor_features)) + 0.001
        
        # Throttle action distribution
        mu_throttle = torch.sigmoid(self.mu_throttle(actor_features))
        sigma_throttle = F.softplus(self.sigma_throttle(actor_features)) + 0.001
        
        # Brake action distribution
        mu_brake = torch.sigmoid(self.mu_brake(actor_features))
        sigma_brake = F.softplus(self.sigma_brake(actor_features)) + 0.001
        
        # Critic value estimation
        value = self.critic_fc(conv_out)
        
        return (mu_steer, sigma_steer), (mu_throttle, sigma_throttle), (mu_brake, sigma_brake), value
    
    def sample_action(self, state):
        (mu_steer, sigma_steer), (mu_throttle, sigma_throttle), (mu_brake, sigma_brake), _ = self(state)
        
        # Create normal distributions for each action
        dist_steer = Normal(mu_steer, sigma_steer)
        dist_throttle = Normal(mu_throttle, sigma_throttle)
        dist_brake = Normal(mu_brake, sigma_brake)
        
        # Sample actions
        steer_action = dist_steer.rsample()
        throttle_action = dist_throttle.rsample()
        brake_action = dist_brake.rsample()
        
        # Clip and transform actions
        steer_action = torch.clamp(steer_action, -1, 1)
        throttle_action = torch.clamp(throttle_action, 0, 1)
        brake_action = torch.clamp(brake_action, 0, 1)
        
        return torch.cat([steer_action, throttle_action, brake_action], dim=-1)

class PPOAgent:
    def __init__(self, input_dims, n_actions, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=3, batch_size=64):
        self.network = ActorCriticNetwork(input_dims, n_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Clipping parameter
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.memory = []
        
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.memory.append((state, action, reward, next_state, done, log_prob, value))
    
    def clear_memory(self):
        self.memory.clear()
    
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.network.sample_action(state)
        
        return action.cpu().numpy()[0]
    
    def learn(self):
        # Separate memory into components
        states = torch.tensor([m[0] for m in self.memory], dtype=torch.float).to(device)
        actions = torch.tensor([m[1] for m in self.memory], dtype=torch.float).to(device)
        rewards = torch.tensor([m[2] for m in self.memory], dtype=torch.float).to(device)
        next_states = torch.tensor([m[3] for m in self.memory], dtype=torch.float).to(device)
        dones = torch.tensor([m[4] for m in self.memory], dtype=torch.float).to(device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        
        # Compute returns and advantages using Generalized Advantage Estimation
        for i in range(len(self.memory)):
            # Compute discounted return
            R = 0
            advantage = 0
            
            # Compute TD error
            _, _, _, curr_value = self.network(states[i].unsqueeze(0))
            _, _, _, next_value = self.network(next_states[i].unsqueeze(0))
            
            curr_value = curr_value.detach()
            next_value = next_value.detach()
            
            if dones[i] == 0:
                R = rewards[i] + self.gamma * next_value
                advantage = rewards[i] + self.gamma * next_value - curr_value
            else:
                R = rewards[i]
                advantage = rewards[i] - curr_value
            
            returns.append(R)
            advantages.append(advantage)
        
        returns = torch.tensor(returns, dtype=torch.float).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform mini-batch updates
        for _ in range(self.epochs):
            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]
                
                # Compute current policy outputs
                (mu_steer, sigma_steer), (mu_throttle, sigma_throttle), (mu_brake, sigma_brake), values = self.network(batch_states)
                
                # Compute policy loss
                policy_loss = self.compute_policy_loss(
                    batch_actions, batch_advantages, 
                    (mu_steer, sigma_steer), 
                    (mu_throttle, sigma_throttle), 
                    (mu_brake, sigma_brake)
                )
                
                # Compute value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)
                self.optimizer.step()
        
        # Clear memory after learning
        self.clear_memory()
    
    def compute_policy_loss(self, actions, advantages, 
                              steer_params, throttle_params, brake_params):
        mu_steer, sigma_steer = steer_params
        mu_throttle, sigma_throttle = throttle_params
        mu_brake, sigma_brake = brake_params
        
        # Create distributions
        dist_steer = Normal(mu_steer, sigma_steer)
        dist_throttle = Normal(mu_throttle, sigma_throttle)
        dist_brake = Normal(mu_brake, sigma_brake)
        
        # Split actions
        steer_actions = actions[:, 0]
        throttle_actions = actions[:, 1]
        brake_actions = actions[:, 2]
        
        # Compute log probabilities
        log_probs_steer = dist_steer.log_prob(steer_actions)
        log_probs_throttle = dist_throttle.log_prob(throttle_actions)
        log_probs_brake = dist_brake.log_prob(brake_actions)
        
        log_probs = log_probs_steer + log_probs_throttle + log_probs_brake
        
        # PPO Clipped Surrogate Objective
        return -torch.mean(log_probs * advantages)

def preprocess_state(state):
    # Convert to float and normalize
    state = state.astype(np.float32) / 255.0
    # Transpose to match PyTorch Conv2d input format
    return np.transpose(state, (2, 0, 1))

def main():
    env = gym.make('CarRacing-v3')
    
    # Training parameters
    num_episodes = 100
    max_steps = 1000
    
    # Initialize agent
    input_dims = (3, 96, 96)  # Image dimensions for CarRacing
    n_actions = 3  # Steer, Throttle, Brake
    agent = PPOAgent(input_dims, n_actions)
    
    # Tracking metrics
    episode_rewards = []
    successful_episodes = 0
    total_training_time = 0
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        
        episode_reward = 0
        done = False
        truncated = False
        
        for step in range(max_steps):
            # Select action
            action = agent.choose_action(state)
            
            # Environment step
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done, None, None)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        # Learning
        agent.learn()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        if episode_reward > 900:  # Consider episode successful if high reward achieved
            successful_episodes += 1
        
        # Print progress
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    total_training_time = time.time() - start_time

    algorithm_name = 'PPO | Car Racing'
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


    
    env.close()



if __name__ == "__main__":
    main()
    
    # Example usage
