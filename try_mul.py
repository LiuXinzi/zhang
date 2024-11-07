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
from stable_baselines3.common.vec_env import SubprocVecEnv

class PPO(nn.Module):
    def __init__(self, n_states, n_actions, device, critic_lr=1e-3, actor_lr=1e-3):
        super(PPO, self).__init__()
        self.device = device
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.1
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
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, n_actions)
        ).to(self.device)
        self._init_weights(self.actor)
        
        self.actor_log_std = nn.Parameter(torch.zeros(n_actions, device=device))
        
        self.critic = nn.Sequential(
            nn.Linear(n_states, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, 1)
        ).to(self.device)
        self._init_weights(self.critic)
        
        # Optimizers
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
    
    def sample_action(self, states):
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        actions = dist.sample()
        actions = torch.clamp(actions, -3, 3)
        self.log_probs.extend(dist.log_prob(actions))
        self.entropies.extend(dist.entropy())
        return actions.cpu().detach().numpy()

    def update(self):
        if not self.rewards:
            return
        rewards = torch.stack([torch.tensor(r, device=self.device, dtype=torch.float32) for r in self.rewards])  # Shape [n, env_num, 1]
        states = torch.stack([torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in env_states]) for env_states in self.states])  # Shape [n+1, env_num, state_dim]
        actions = torch.stack([torch.tensor(a, device=self.device, dtype=torch.float32) for a in self.actions])  # Shape [n, env_num, action_dim]
        log_probs_old = torch.stack(self.log_probs).detach()  # Shape [n, env_num, 1]
        entropies = torch.stack(self.entropies).detach()  # Shape [n, env_num, 1]


        returns = torch.zeros_like(rewards)  # Shape [n, env_num, 1]
        returns[-1] = rewards[-1]
        if self.truncated:
            with torch.no_grad():
                last_value = self.critic(states[-1]).detach().squeeze(-1)  # Shape [env_num]
            returns[-1] += self.gamma * last_value.unsqueeze(-1)

        # 逐步计算 returns，沿着时间维度反向计算
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]

        # 将 states 除去最后一个时间步以对齐 actions 和 rewards 维度
        states = states[:-1]  # 形状变为 [n, env_num, state_dim]

        # 将数据扁平化以进行批量更新
        flat_states = states.view(-1, states.shape[-1])  # Shape [n * env_num, state_dim]
        flat_returns = returns.view(-1)  # Shape [n * env_num]
        flat_actions = actions.view(-1, actions.shape[-1])  # Shape [n * env_num, action_dim]
        flat_log_probs_old = log_probs_old.view(-1)  # Shape [n * env_num]
        flat_entropies = entropies.view(-1)  # Shape [n * env_num]

        # 更新 Critic
        for _ in range(self.train_V_iters):
            V_pred = self.critic(flat_states).squeeze(-1)  # Shape [n * env_num]
            critic_loss = F.smooth_l1_loss(V_pred, flat_returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 计算优势 advantages
        with torch.no_grad():
            V_pred = self.critic(flat_states).detach().squeeze(-1)  # Shape [n * env_num]
        td_errors = flat_returns - V_pred

        advantages = td_errors.view(rewards.shape[0], rewards.shape[1])  # Reshape to [n, env_num] for GAE calculation
        for t in reversed(range(len(rewards) - 1)):
            advantages[t] += self.gamma * self.lam * advantages[t + 1]

        # 优势归一化
        flat_advantages = advantages.view(-1)  # Shape [n * env_num]
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + self.eps)

        # 更新 Actor
        for _ in range(self.train_A_iters):
            mu = self.actor(flat_states)  # Shape [n * env_num, action_dim]
            sigma = torch.exp(self.actor_log_std).expand_as(mu)
            dists = Normal(mu, sigma)
            log_probs = dists.log_prob(flat_actions).sum(dim=-1)  # Shape [n * env_num]
            r_theta = torch.exp(log_probs - flat_log_probs_old)
            clipped_r_theta = torch.clamp(r_theta, 1 - self.epsilon, 1 + self.epsilon)

            # 使用 PPO 剪辑策略计算 actor_loss
            actor_loss = -torch.min(r_theta * flat_advantages.detach(), clipped_r_theta * flat_advantages.detach()).mean()
            actor_loss -= self.entropy_coef * flat_entropies.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 计算 KL 散度以提前停止
            approx_kl = (flat_log_probs_old - log_probs).mean().item()
            if approx_kl > self.KL_threshold:
                break

        # 重置数据缓存
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

def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == '__main__':
    # 其他的主要程序逻辑放在这里
    num_envs = 3
    env_id = "MFGIP-v0"
    envs = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)])
    
    total_num_episodes = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    obs_space_dims = envs.observation_space.shape[0]
    action_space_dims = envs.action_space.shape[0]

    agent = PPO(obs_space_dims, action_space_dims, device)

    for episode in range(total_num_episodes):
        obs = envs.reset()
        done = False
        agent.states.append(obs)
        
        while not done:
            actions = agent.sample_action(obs)
            
            obs, rewards, dones, truns= envs.step(actions)
            # import ipdb;ipdb.set_trace()
            agent.actions.append(actions)
            agent.rewards.append(rewards)
            agent.states.append(obs)
            
            if any(dones):
                done = True
                # agent.truncated = any([info.get("TimeLimit.truncated", False) for info in infos])
        
        agent.update()
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward: ")
          

    # envs.close() 
    print("Training completed.")
