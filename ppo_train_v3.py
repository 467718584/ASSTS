"""
PPO强化学习训练 - 数据集分配版
训练/验证/测试 比例: 7:1:2
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
import random

sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'

# 数据集分配比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 超参数
ACTOR_LR = 5e-5
CRITIC_LR = 1e-4
GAMMA = 0.995
LAMBDA = 0.98
EPS_CLIP = 0.2
K_EPOCHS = 10
UPDATE_INTERVAL = 128
MAX_EPISODES = 1000
WINDOW_SIZE = 10
HIDDEN_DIM = 256
ENTROPY_COEF = 0.03
VALUE_COEF = 0.5
LR_DECAY = 0.98

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


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
                shaped_rewards.append(r * 2.0)
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
            nn.Dropout(0.1),
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
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def split_dataset(stock_codes, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """划分训练/验证/测试数据集"""
    random.shuffle(stock_codes)
    n = len(stock_codes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_stocks = stock_codes[:n_train]
    val_stocks = stock_codes[n_train:n_train+n_val]
    test_stocks = stock_codes[n_train+n_val:]
    
    return train_stocks, val_stocks, test_stocks


class StockTradingEnvSplit(StockTradingEnv):
    """支持数据集划分的环境"""
    
    def __init__(self, stock_data_dir, stock_list, window_size=10, initial_capital=10000):
        self.stock_data_dir = stock_data_dir
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.stock_codes = stock_list  # 使用指定的股票列表
        
        if len(self.stock_codes) == 0:
            raise ValueError("No stocks in list")
        
        self.action_space = 2
        self.observation_space = window_size * 5
        self.stop_loss = -0.03
        self.transaction_fee = 0.001
        
        self.reset()


def train_model(pool_path, model_name):
    """训练单个模型"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 获取所有股票并划分
    all_stocks = [d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))]
    all_stocks = [s for s in all_stocks if len([f for f in os.listdir(os.path.join(pool_path, s)) if f.endswith('.csv')]) >= 3]
    
    train_stocks, val_stocks, test_stocks = split_dataset(all_stocks)
    
    print(f"    数据集分配: 训练={len(train_stocks)}, 验证={len(val_stocks)}, 测试={len(test_stocks)}")
    
    # 训练集
    env = StockTradingEnvSplit(pool_path, train_stocks, window_size=WINDOW_SIZE)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    agent = PPOAgent(state_dim, action_dim)
    
    best_val_profit = -float('inf')
    best_agent = None
    patience = 0
    max_patience = 80
    
    for episode in range(MAX_EPISODES):
        state = env.reset()
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.add(state, action, reward, done, log_prob, value)
            
            if len(agent.memory.states) >= UPDATE_INTERVAL:
                agent.update()
            
            state = next_state
        
        # 验证
        if episode % 20 == 0 and len(val_stocks) > 0:
            val_env = StockTradingEnvSplit(pool_path, val_stocks, window_size=WINDOW_SIZE)
            val_profits = []
            for _ in range(20):
                state = val_env.reset()
                done = False
                while not done:
                    action, _, _ = agent.select_action(state, training=False)
                    state, reward, done, info = val_env.step(action)
                val_profits.append(info.get('profit', 0))
            
            val_profit = np.mean(val_profits)
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                best_agent = PPOAgent(state_dim, action_dim)
                best_agent.actor.load_state_dict(agent.actor.state_dict())
                best_agent.critic.load_state_dict(agent.critic.state_dict())
                patience = 0
            else:
                patience += 1
            
            if patience >= max_patience:
                break
    
    # 测试
    if best_agent is None:
        best_agent = agent
    
    test_env = StockTradingEnvSplit(pool_path, test_stocks, window_size=WINDOW_SIZE)
    test_profits = []
    for _ in range(50):
        state = test_env.reset()
        done = False
        while not done:
            action, _, _ = best_agent.select_action(state, training=False)
            state, reward, done, info = test_env.step(action)
        test_profits.append(info.get('profit', 0))
    
    # 训练集统计
    train_env = StockTradingEnvSplit(pool_path, train_stocks, window_size=WINDOW_SIZE)
    train_profits = []
    for _ in range(50):
        state = train_env.reset()
        done = False
        while not done:
            action, _, _ = best_agent.select_action(state, training=False)
            state, reward, done, info = train_env.step(action)
        train_profits.append(info.get('profit', 0))
    
    result = {
        'model': model_name,
        'total_stocks': len(all_stocks),
        'train_count': len(train_stocks),
        'val_count': len(val_stocks),
        'test_count': len(test_stocks),
        'train_best_profit': float(max(train_profits)),
        'train_avg_profit': float(np.mean(train_profits)),
        'train_positive_rate': float(len([p for p in train_profits if p > 0]) / len(train_profits)),
        'test_best_profit': float(max(test_profits)),
        'test_avg_profit': float(np.mean(test_profits)),
        'test_positive_rate': float(len([p for p in test_profits if p > 0]) / len(test_profits)),
    }
    
    print(f"    训练: 最佳{max(train_profits)*100:.2f}%, 平均{np.mean(train_profits)*100:.2f}%")
    print(f"    测试: 最佳{max(test_profits)*100:.2f}%, 平均{np.mean(test_profits)*100:.2f}%")
    
    return result, best_agent


def main():
    print("="*60)
    print("PPO训练 (严格3天数据 + 数据集划分)")
    print(f"数据集分配: 训练{TRAIN_RATIO*100:.0f}%: 验证{VAL_RATIO*100:.0f}%: 测试{TEST_RATIO*100:.0f}%")
    print("="*60)
    
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
            continue
        
        stock_count = len([d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))])
        
        print(f"\n{'='*50}")
        print(f"模型: {model_name}, 股票数: {stock_count}")
        print(f"{'='*50}")
        
        if stock_count < 10:
            print(f"  股票数量不足，跳过")
            continue
        
        result, agent = train_model(pool_path, model_name)
        
        if result:
            model_path = os.path.join(MODEL_OUTPUT_DIR, f'ppo_{model_name}_v3.pth')
            agent.save(model_path)
            all_results.append(result)
    
    # 保存结果
    output_file = os.path.join(MODEL_OUTPUT_DIR, 'ppo_v3_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    all_results.sort(key=lambda x: x['test_best_profit'], reverse=True)
    
    print(f"\n| 排名 | 模型 | 总股票 | 训练 | 验证 | 测试 | 测试最佳 | 测试平均 | 正收益 |")
    print(f"|:---:|:---:|:---:|:---:|:---:|:---:|--------:|--------:|:---:|")
    
    for i, r in enumerate(all_results):
        print(f"| {i+1} | {r['model']} | {r['total_stocks']} | {r['train_count']} | {r['val_count']} | {r['test_count']} | {r['test_best_profit']*100:.2f}% | {r['test_avg_profit']*100:.2f}% | {r['test_positive_rate']*100:.1f}% |")
    
    print(f"\n结果已保存: {output_file}")


if __name__ == '__main__':
    main()
