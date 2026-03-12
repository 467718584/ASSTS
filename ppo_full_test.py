"""
全量PPO模型测试脚本
对6个模型分别训练并评估
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

sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'
MIN_DATA_DIR = '/home/zzy/project/ASSTS/data_min'

# 6个模型配置
MODELS = {
    '30d_2s3e': {'file': '0d_30d_14f_2s3e_BCE_0005-003_AUC_05437.pth', 'length': 30},
    '30d_2s3h': {'file': '0d_30d_14f_2s3h_BCE_0005-003_AUC_05844.pth', 'length': 30},
    '60d_2s3e': {'file': '0d_60d_14f_2s3e_BCE_0005-003_AUC_05343.pth', 'length': 60},
    '60d_2s3h': {'file': '0d_60d_14f_2s3h_BCE_0005-003_AUC_05736.pth', 'length': 60},
    '120d_2s3e': {'file': '0d_120d_14f_2s3e_BCE_0005-003_AUC_05434.pth', 'length': 120},
    '120d_2s3h': {'file': '0d_120d_14f_2s3h_BCE_0005-003_AUC_05843.pth', 'length': 120},
}

# 超参数
ACTOR_LR = 3e-5
CRITIC_LR = 1e-4
GAMMA = 0.995
LAMBDA = 0.98
EPS_CLIP = 0.1
K_EPOCHS = 20
UPDATE_INTERVAL = 512
MAX_EPISODES = 300  # 减少轮数加快测试
WINDOW_SIZE = 10
HIDDEN_DIM = 256
ENTROPY_COEF = 0.02
VALUE_COEF = 1.0
LR_DECAY = 0.995


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
        shaped_rewards = []
        for r in rewards:
            if r > 0:
                shaped_rewards.append(r * 2)
            elif r < -0.02:
                shaped_rewards.append(r * 0.5)
            else:
                shaped_rewards.append(r)
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
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
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
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM)
        self.critic = Critic(state_dim, HIDDEN_DIM)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.scheduler_actor = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=LR_DECAY)
        self.scheduler_critic = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=LR_DECAY)
        
        self.gamma = GAMMA
        self.lambda_ = LAMBDA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        self.entropy_coef = ENTROPY_COEF
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
        
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item()


def load_stock_pool(model_name):
    """加载股票池"""
    pool_path = os.path.join(POOL_DIR, f'ppo_pool_{model_name}_min1')
    if not os.path.exists(pool_path):
        print(f"股票池不存在: {pool_path}")
        return None
    
    # 获取所有股票
    stock_codes = [d for d in os.listdir(pool_path) 
                  if os.path.isdir(os.path.join(pool_path, d))]
    return pool_path, stock_codes


def train_and_evaluate(model_name, pool_info):
    """训练并评估单个模型"""
    pool_path, stock_codes = pool_info
    
    if len(stock_codes) < 10:
        print(f"  股票数量不足: {len(stock_codes)}")
        return None
    
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"股票池: {pool_path}")
    print(f"股票数量: {len(stock_codes)}")
    print(f"{'='*60}")
    
    # 创建环境
    env = StockTradingEnv(pool_path, window_size=WINDOW_SIZE)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    # 创建Agent
    agent = PPOAgent(state_dim, action_dim)
    
    # 训练
    episode_profits = []
    episode_rewards = []
    
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
    
    # 评估
    results = {
        'model': model_name,
        'episodes': MAX_EPISODES,
        'stock_count': len(stock_codes),
        'best_profit': float(max(episode_profits)),
        'avg_profit': float(np.mean(episode_profits)),
        'positive_rate': float(len([p for p in episode_profits if p > 0]) / len(episode_profits)),
        'avg_reward': float(np.mean(episode_rewards)),
    }
    
    print(f"\n结果:")
    print(f"  最佳收益: {results['best_profit']*100:.2f}%")
    print(f"  平均收益: {results['avg_profit']*100:.2f}%")
    print(f"  正收益比例: {results['positive_rate']*100:.1f}%")
    
    return results


def main():
    print("="*60)
    print("全量PPO模型测试")
    print("="*60)
    
    all_results = []
    
    for model_key, model_info in MODELS.items():
        print(f"\n处理模型: {model_key}")
        
        # 加载股票池
        pool_info = load_stock_pool(model_key)
        if pool_info is None:
            print(f"  跳过 {model_key} (无股票池)")
            continue
        
        # 训练评估
        result = train_and_evaluate(model_key, pool_info)
        if result:
            all_results.append(result)
    
    # 保存结果
    output_file = os.path.join(MODEL_OUTPUT_DIR, 'ppo_test_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("所有模型测试完成!")
    print(f"{'='*60}")
    
    # 排序找出最佳模型
    all_results.sort(key=lambda x: x['best_profit'], reverse=True)
    
    print("\n模型排名 (按最佳收益):")
    for i, r in enumerate(all_results):
        print(f"{i+1}. {r['model']}: 最佳{r['best_profit']*100:.2f}%, 平均{r['avg_profit']*100:.2f}%")
    
    print(f"\n最佳模型: {all_results[0]['model']}")
    print(f"结果已保存: {output_file}")
    
    return all_results


if __name__ == '__main__':
    main()
