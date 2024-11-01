# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 23:49:40 2024

@author: YAKE
"""

import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import time
import gymnasium as gym

class PPO(nn.Module):
    def __init__(self, n_states, n_actions, device, critic_lr=1e-3, actor_lr=1e-3):
        super(PPO, self).__init__()
        self.device = device
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        # TODO: try using a learning rate scheduler to dynamically adjust the learning rate based on the training process
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.1  # 0.1~0.2
        self.KL_threshold = 1e-2
        self.entropy_coef = 0.05
        self.train_V_iters = 10
        self.train_A_iters = 10
        self.dropout = 0.3
        self.eps = np.finfo(np.float32).eps.item()
    
        self.rewards = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.entropies = []
        self.truncated = False
        
        num_h1 = 128
        num_h2 = 64
        self.actor = nn.Sequential(
            nn.Linear(n_states, num_h1),
            nn.ReLU(),
            #nn.Dropout(self.dropout), # Try removing the dropout layer
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            #nn.Dropout(self.dropout),
            nn.Linear(num_h2, n_actions)
            ).to(self.device)
        self._init_weights(self.actor)
        
        self.actor_log_std = nn.Parameter(torch.zeros(n_actions, device=device))
        
        self.critic = nn.Sequential(
            nn.Linear(n_states, num_h1),
            nn.ReLU(),
            #nn.Dropout(self.dropout),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            #nn.Dropout(self.dropout),
            nn.Linear(num_h2, 1)
            ).to(self.device)
        self._init_weights(self.critic)
        
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-4)
        self.actor_optimizer = torch.optim.AdamW(list(self.actor.parameters()) + [self.actor_log_std], lr=self.actor_lr, weight_decay=1e-4)
        
    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        mean = self.actor(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        value = self.critic(x)
        return mean, std, value
    
    def sample_action(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -3, 3)
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        return action.cpu().detach().numpy()

    def update(self):
        if not self.rewards:
            return
        
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32)
        states = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in self.states])
        actions = torch.tensor(self.actions, device=self.device, dtype=torch.float32)
        log_probs_old = torch.stack(self.log_probs).squeeze(-1).detach()
        entropies = torch.stack(self.entropies).squeeze(-1).detach()

        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        if self.truncated:
            with torch.no_grad():
                last_value = self.critic(states[-1]).detach().squeeze(-1)
            returns[-1] += self.gamma * last_value
        
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        
        for _ in range(self.train_V_iters):
            V_pred = self.critic(states[:-1])
            critic_loss =  F.smooth_l1_loss(V_pred.squeeze(-1), returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        with torch.no_grad():
            V_pred = self.critic(states).detach().squeeze(-1)
        td_errors = rewards + self.gamma * V_pred[1:] - V_pred[:-1]
        
        advantages = torch.zeros_like(rewards)
        advantages[-1] = td_errors[-1]
        for t in reversed(range(len(rewards) - 1)):
            advantages[t] = td_errors[t] + self.gamma * self.lam * advantages[t + 1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        
        for _ in range(self.train_A_iters):
            mu = self.actor(states[:-1])
            sigma = torch.exp(self.actor_log_std).expand_as(mu)
            dists = Normal(mu,sigma)
            log_probs = dists.log_prob(actions).squeeze()
            r_theta = torch.exp(log_probs - log_probs_old)
            clipped = torch.where(advantages > 0,
                                  torch.min(r_theta, torch.tensor(1 + self.epsilon, dtype=torch.float32, device=self.device)),
                                  torch.max(r_theta, torch.tensor(1 - self.epsilon, dtype=torch.float32, device=self.device))
                                  )
            actor_loss = -(clipped * advantages.detach()).mean()
            actor_loss -= self.entropy_coef * entropies.mean()
        
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            approx_kl = (log_probs_old - log_probs).mean().item()
            if approx_kl > self.KL_threshold:
                break
            
        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []
        self.entropies = []
        self.truncated = False
        
from gymnasium.envs.registration import register
register(
    id='MFGIP-v0',
    entry_point='Inverted_Pendulum:MFG_InvertedPendulum',
    max_episode_steps=1000
)

env = gym.make("MFGIP-v0")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)

total_num_episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

obs_space_dims = env.observation_space.shape[0]
action_space_dims = env.action_space.shape[0]

rewards_over_seeds = []
start_time = time.time()
initial_entropy_coef = 0.05
final_entropy_coef = 0.01
decay_rate = (initial_entropy_coef - final_entropy_coef) / total_num_episodes

for seed in [1, 2, 3, 5, 8]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 
    
    reward_over_episodes = []
    agent = PPO(obs_space_dims, action_space_dims, device)
    
    for episode in range(total_num_episodes):
        current_entropy_coef = max(final_entropy_coef, initial_entropy_coef - episode * decay_rate)
        agent.entropy_coef = current_entropy_coef
        
        obs, _ = wrapped_env.reset(seed=seed)
        agent.states.append(obs)
        done = False
        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.states.append(obs)
            done = terminated or truncated
        
        reward_over_episodes.append(info['episode']['r']) if 'episode' in info else 0
        if truncated:
            agent.truncated = True
            
        agent.update()
        
        if episode % 10 == 0 and 'episode' in info:
            print(f"Seed {seed},  Episode {episode}: Total Reward: {info['episode']['r'] if 'episode' in info else 'N/A'}, Length: {info['episode']['l'] if 'episode' in info else 'N/A'}")
            
        
    rewards_over_seeds.append(reward_over_episodes)

#env.close()
duration = time.time() - start_time
print(f"Duration: {duration:.4f} seconds")
plt.figure()
for idx, rewards in enumerate(rewards_over_seeds):
    plt.plot(rewards, label=f'Reward per episode for seed {idx + 1}')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Rewards over Episodes for Different Seeds')
plt.legend()
plt.grid(True)
plt.show()