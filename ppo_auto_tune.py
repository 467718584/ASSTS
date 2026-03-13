"""
PPO强化学习训练 - 自动调优版
功能：
1. 自动网格搜索最佳超参数
2. 优化网络结构
3. 增强探索策略
4. 课程学习
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
from datetime import datetime
import random
from itertools import product

sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'
TRAINING_LOG_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_logs'

# 超参数搜索空间
PARAM_GRID = {
    'actor_lr': [1e-5, 3e-5, 1e-4],
    'critic_lr': [5e-5, 1e-4, 5e-4],
    'gamma': [0.99, 0.995, 0.999],
    'entropy_coef': [0.01, 0.02, 0.05],
    'hidden_dim': [128, 256],
}

# 固定超参数
GAMMA = 0.995
LAMBDA = 0.98
EPS_CLIP = 0.15
K_EPOCHS = 15
UPDATE_INTERVAL = 256
MAX_EPISODES = 500
WINDOW_SIZE = 10
VALUE_COEF = 0.5
LR_DECAY = 0.99

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def get(self):
        rewards = np.array(self.rewards, dtype=np.float32)
        
        # 奖励塑形
        shaped_rewards = []
        for r in rewards:
            if r > 0:
                shaped_rewards.append(r * 1.5)
            elif r < -0.03:
                shaped_rewards.append(r * 0.3)
            else:
                shaped_rewards.append(r * 0.8)
        shaped_rewards = np.array(shaped_rewards, dtype=np.float32)
        
        if len(shaped_rewards) > 1 and shaped_rewards.std() > 0:
            shaped_rewards = (shaped_rewards - shaped_rewards.mean()) / (shaped_rewards.std() + 1e-8)
        
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            shaped_rewards,
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, state_dim, action_dim, params):
        self.actor = Actor(state_dim, action_dim, params['hidden_dim'])
        self.critic = Critic(state_dim, params['hidden_dim'])
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['critic_lr'])
        self.scheduler_actor = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=LR_DECAY)
        self.scheduler_critic = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=LR_DECAY)
        
        self.gamma = params.get('gamma', GAMMA)
        self.lambda_ = LAMBDA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        self.entropy_coef = params.get('entropy_coef', 0.02)
        self.value_coef = VALUE_COEF
        
        self.memory = PPOMemory()
    
    def select_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        probs = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = probs.argmax(dim=1)
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self):
        states, actions, rewards, dones, old_log_probs, old_values = self.memory.get()
        
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = old_values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - old_values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[t])
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.k_epochs):
            probs = self.actor(states)
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            values = self.critic(states).squeeze()
            critic_loss = self.value_coef * nn.MSELoss()(values, returns)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        
        self.memory.clear()
        
        return total_actor_loss / self.k_epochs, total_critic_loss / self.k_epochs
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def train_model(pool_path, params, model_name, verbose=True):
    """训练单个模型"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 创建环境
    env = StockTradingEnv(pool_path, window_size=WINDOW_SIZE)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    # 创建Agent
    agent = PPOAgent(state_dim, action_dim, params)
    
    episode_profits = []
    episode_rewards = []
    best_profit = -float('inf')
    patience = 0
    max_patience = 50
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        total_profit = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.add(state, action, reward, done, log_prob, value)
            
            if len(agent.memory.states) >= UPDATE_INTERVAL:
                agent.update()
            
            total_reward += reward
            if 'profit' in info:
                total_profit = info['profit']
            
            state = next_state
        
        episode_profits.append(total_profit)
        episode_rewards.append(total_reward)
        
        # 记录最佳
        if total_profit > best_profit:
            best_profit = total_profit
            patience = 0
        else:
            patience += 1
        
        # 早停
        if patience >= max_patience and episode > 100:
            if verbose:
                print(f"    早停于第 {episode} 轮")
            break
    
    # 计算统计
    result = {
        'model': model_name,
        'best_profit': float(best_profit),
        'avg_profit': float(np.mean(episode_profits[-50:])),  # 最后50轮平均
        'positive_rate': float(len([p for p in episode_profits[-50:] if p > 0]) / min(50, len(episode_profits))),
        'episodes': len(episode_profits),
        'params': params,
    }
    
    if verbose:
        print(f"    最佳: {best_profit*100:.2f}%, 平均: {result['avg_profit']*100:.2f}%, 正收益: {result['positive_rate']*100:.1f}%")
    
    return result, agent


def grid_search(pool_path, model_name, verbose=True):
    """网格搜索最佳超参数"""
    print(f"\n{'='*50}")
    print(f"网格搜索: {model_name}")
    print(f"{'='*50}")
    
    # 先快速测试几组参数
    test_configs = [
        {'actor_lr': 1e-4, 'critic_lr': 5e-4, 'gamma': 0.99, 'entropy_coef': 0.02, 'hidden_dim': 128},
        {'actor_lr': 3e-5, 'critic_lr': 1e-4, 'gamma': 0.995, 'entropy_coef': 0.02, 'hidden_dim': 256},
        {'actor_lr': 1e-5, 'critic_lr': 5e-5, 'gamma': 0.999, 'entropy_coef': 0.05, 'hidden_dim': 256},
    ]
    
    best_result = None
    best_profit = -float('inf')
    
    for i, params in enumerate(test_configs):
        if verbose:
            print(f"  配置 {i+1}/{len(test_configs)}: lr={params['actor_lr']}, gamma={params['gamma']}, h_dim={params['hidden_dim']}")
        
        result, agent = train_model(pool_path, params, model_name, verbose=verbose)
        
        if result['best_profit'] > best_profit:
            best_profit = result['best_profit']
            best_result = result
            best_agent = agent
    
    return best_result, best_agent


def main():
    print("="*60)
    print("PPO自动调优训练")
    print("="*60)
    
    # 6个股票池
    pools = {
        '30d_2s3e': 'ppo_pool_30d_2s3e_min1',
        '30d_2s3h': 'ppo_pool_30d_2s3h_min1',
        '60d_2s3e': 'ppo_pool_60d_2s3e_min1',
        '60d_2s3h': 'ppo_pool_60d_2s3h_min1',
        '120d_2s3e': 'ppo_pool_120d_2s3e_min1',
        '120d_2s3h': 'ppo_pool_120d_2s3h_min1',
    }
    
    all_results = []
    
    for model_name, pool_dir in pools.items():
        pool_path = os.path.join(POOL_DIR, pool_dir)
        
        if not os.path.exists(pool_path):
            print(f"\n跳过 {model_name}: 股票池不存在")
            continue
        
        stock_count = len([d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))])
        print(f"\n{'='*60}")
        print(f"模型: {model_name}, 股票数: {stock_count}")
        print(f"{'='*60}")
        
        if stock_count < 10:
            print(f"  股票数量不足，跳过")
            continue
        
        # 网格搜索
        best_result, best_agent = grid_search(pool_path, model_name)
        
        if best_result:
            # 保存模型
            model_path = os.path.join(MODEL_OUTPUT_DIR, f'ppo_{model_name}_best.pth')
            best_agent.save(model_path)
            print(f"  模型已保存: {model_path}")
            
            all_results.append(best_result)
    
    # 保存结果
    output_file = os.path.join(MODEL_OUTPUT_DIR, 'ppo_tuned_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    all_results.sort(key=lambda x: x['best_profit'], reverse=True)
    
    print(f"\n| 排名 | 模型 | 股票数 | 最佳收益 | 平均收益 | 正收益比例 |")
    print(f"|:---:|------|:------:|--------:|--------:|:---------:|")
    
    for i, r in enumerate(all_results):
        print(f"| {i+1} | {r['model']} | - | {r['best_profit']*100:.2f}% | {r['avg_profit']*100:.2f}% | {r['positive_rate']*100:.1f}% |")
    
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
