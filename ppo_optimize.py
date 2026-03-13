"""
PPO优化实验 - 穷举式超参数调优
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
from collections import deque

sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# 默认超参数（baseline）
DEFAULT_PARAMS = {
    'ACTOR_LR': 5e-5,
    'CRITIC_LR': 1e-4,
    'GAMMA': 0.995,
    'LAMBDA': 0.98,
    'EPS_CLIP': 0.2,
    'K_EPOCHS': 10,
    'UPDATE_INTERVAL': 64,
    'MAX_EPISODES': 2000,
    'WINDOW_SIZE': 10,
    'HIDDEN_DIM': 256,
    'ENTROPY_COEF': 0.03,
    'VALUE_COEF': 0.5,
    'LR_DECAY': 0.98,
}

# 动态导入
def get_env():
    from ppo_env import StockTradingEnv
    from ppo_detailed import StockTradingEnvDetailed
    return StockTradingEnv, StockTradingEnvDetailed

StockTradingEnv, StockTradingEnvDetailed = get_env()

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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
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
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, state_dim, action_dim, params):
        self.actor = Actor(state_dim, action_dim, params['HIDDEN_DIM'])
        self.critic = Critic(state_dim, params['HIDDEN_DIM'])
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['ACTOR_LR'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['CRITIC_LR'])
        
        self.gamma = params['GAMMA']
        self.lambda_ = params['LAMBDA']
        self.eps_clip = params['EPS_CLIP']
        self.k_epochs = params['K_EPOCHS']
        self.entropy_coef = params['ENTROPY_COEF']
        self.value_coef = params['VALUE_COEF']
        
        self.MSELoss = nn.MSELoss()
    
    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        
        if training:
            action = dist.sample()
        else:
            action = probs.argmax(dim=1)
        
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, memory):
        states, actions, rewards, dones, old_log_probs, old_values = memory.get()
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.FloatTensor(old_log_probs)
        old_values = torch.FloatTensor(old_values)
        
        for _ in range(self.k_epochs):
            probs = self.actor(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            new_values = self.critic(states)
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = rewards - old_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = self.MSELoss(new_values.squeeze(), rewards)
            entropy_loss = -dist.entropy().mean()
            
            self.actor_optimizer.zero_grad()
            total_actor_loss = actor_loss + self.entropy_coef * entropy_loss
            total_actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            total_critic_loss = self.value_coef * critic_loss
            total_critic_loss.backward()
            self.critic_optimizer.step()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(stock_codes, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    n = len(stock_codes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    random.shuffle(stock_codes)
    train_stocks = stock_codes[:n_train]
    val_stocks = stock_codes[n_train:n_train+n_val]
    test_stocks = stock_codes[n_train+n_val:]
    
    return train_stocks, val_stocks, test_stocks


def train_and_evaluate(pool_path, model_name, params, verbose=False):
    set_seed(42)
    
    all_stocks = [d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))]
    if len(all_stocks) < 10:
        return None
    
    train_stocks, val_stocks, test_stocks = split_dataset(all_stocks)
    
    env = StockTradingEnvDetailed(pool_path, train_stocks, window_size=params['WINDOW_SIZE'])
    state_dim = env.observation_space
    action_dim = env.action_space
    
    agent = PPOAgent(state_dim, action_dim, params)
    
    memory = PPOMemory()
    episode_rewards = []
    
    for episode in range(params['MAX_EPISODES']):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            memory.add(state, action, reward, done, log_prob, value)
            total_reward += reward
            state = next_state
            
            if done:
                if 'profit' in info:
                    episode_rewards.append(info['profit'])
        
        if (episode + 1) % params['UPDATE_INTERVAL'] == 0:
            agent.update(memory)
            memory.clear()
    
    # 评估 - 遍历所有股票
    def evaluate_set(stock_list, name):
        if len(stock_list) == 0:
            return [], []
        
        results = []
        profits = []
        
        for stock_code in stock_list:
            test_env = StockTradingEnvDetailed(pool_path, [stock_code], window_size=params['WINDOW_SIZE'])
            state = test_env.reset()
            done = False
            while not done:
                action, _, _ = agent.select_action(state, training=False)
                next_state, reward, done, info = test_env.step(action)
                state = next_state
            
            if 'profit' in info:
                results.append({
                    'stock_code': test_env.current_stock,
                    'buy_price': test_env.buy_price,
                    'sell_price': test_env.current_price if test_env.position == 0 else test_env.sell_price,
                    'profit': info['profit']
                })
                profits.append(info['profit'])
        
        return results, profits
    
    train_results, train_profits = evaluate_set(train_stocks, "训练集")
    val_results, val_profits = evaluate_set(val_stocks, "验证集")
    test_results, test_profits = evaluate_set(test_stocks, "测试集")
    
    # 计算统计
    def calc_stats(profits, name):
        if len(profits) == 0:
            return 0, 0
        avg = np.mean(profits)
        win_rate = np.sum(np.array(profits) > 0) / len(profits) * 100
        return avg, win_rate
    
    train_avg, train_win = calc_stats(train_profits, "Train")
    val_avg, val_win = calc_stats(val_profits, "Val")
    test_avg, test_win = calc_stats(test_profits, "Test")
    
    if verbose:
        print(f"\n{model_name} 评估结果:")
        print(f"  Train: 平均={train_avg*100:.2f}%, 胜率={train_win:.1f}%")
        print(f"  Val:   平均={val_avg*100:.2f}%, 胜率={val_win:.1f}%")
        print(f"  Test:  平均={test_avg*100:.2f}%, 胜率={test_win:.1f}%")
    
    return {
        'model': model_name,
        'params': params,
        'train': {'count': len(train_stocks), 'avg': train_avg, 'win_rate': train_win, 'results': train_results},
        'val': {'count': len(val_stocks), 'avg': val_avg, 'win_rate': val_win, 'results': val_results},
        'test': {'count': len(test_stocks), 'avg': test_avg, 'win_rate': test_win, 'results': test_results},
    }


def main():
    print("="*60)
    print("PPO超参数优化实验")
    print("="*60)
    
    pools = {
        '30d_2s3e': 'ppo_pool_30d_2s3e_min1',
        '30d_2s3h': 'ppo_pool_30d_2s3h_min1',
        '60d_2s3e': 'ppo_pool_60d_2s3e_min1',
        '60d_2s3h': 'ppo_pool_60d_2s3h_min1',
        '120d_2s3e': 'ppo_pool_120d_2s3e_min1',
        '120d_2s3h': 'ppo_pool_120d_2s3h_min1',
    }
    
    # 实验参数组合 - 精简版用于快速测试
    param_combinations = [
        {'name': 'baseline', 'ACTOR_LR': 5e-5, 'CRITIC_LR': 1e-4, 'GAMMA': 0.995, 'ENTROPY_COEF': 0.03, 'HIDDEN_DIM': 256},
        {'name': 'lr_high', 'ACTOR_LR': 1e-4, 'CRITIC_LR': 2e-4, 'GAMMA': 0.995, 'ENTROPY_COEF': 0.03, 'HIDDEN_DIM': 256},
        {'name': 'lr_low', 'ACTOR_LR': 1e-5, 'CRITIC_LR': 5e-5, 'GAMMA': 0.995, 'ENTROPY_COEF': 0.03, 'HIDDEN_DIM': 256},
        {'name': 'gamma_high', 'ACTOR_LR': 5e-5, 'CRITIC_LR': 1e-4, 'GAMMA': 0.999, 'ENTROPY_COEF': 0.03, 'HIDDEN_DIM': 256},
        {'name': 'gamma_low', 'ACTOR_LR': 5e-5, 'CRITIC_LR': 1e-4, 'GAMMA': 0.99, 'ENTROPY_COEF': 0.03, 'HIDDEN_DIM': 256},
        {'name': 'entropy_low', 'ACTOR_LR': 5e-5, 'CRITIC_LR': 1e-4, 'GAMMA': 0.995, 'ENTROPY_COEF': 0.01, 'HIDDEN_DIM': 256},
        {'name': 'entropy_high', 'ACTOR_LR': 5e-5, 'CRITIC_LR': 1e-4, 'GAMMA': 0.995, 'ENTROPY_COEF': 0.05, 'HIDDEN_DIM': 256},
    ]
    
    all_experiments = []
    
    for model_name, pool_dir in pools.items():
        pool_path = os.path.join(POOL_DIR, pool_dir)
        
        if not os.path.exists(pool_path):
            continue
        
        stock_count = len([d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))])
        
        if stock_count < 10:
            continue
        
        print(f"\n{'='*40}")
        print(f"模型: {model_name} ({stock_count}只股票)")
        print(f"{'='*40}")
        
        model_experiments = []
        
        for exp in param_combinations:
            params = DEFAULT_PARAMS.copy()
            params.update({k: v for k, v in exp.items() if k != 'name'})
            exp_name = exp['name']
            
            print(f"\n  实验: {exp_name}...", end=" ", flush=True)
            
            result = train_and_evaluate(pool_path, model_name, params, verbose=False)
            
            if result:
                model_experiments.append({
                    'exp_name': exp_name,
                    'test_avg': result['test']['avg'],
                    'test_win_rate': result['test']['win_rate'],
                    'val_avg': result['val']['avg'],
                    'val_win_rate': result['val']['win_rate'],
                })
                print(f"Test: {result['test']['avg']*100:.2f}%, {result['test']['win_rate']:.1f}%")
        
        # 找最佳实验
        if model_experiments:
            best_exp = max(model_experiments, key=lambda x: x['test_avg'])
            all_experiments.append({
                'model': model_name,
                'best_exp': best_exp['exp_name'],
                'best_test_avg': best_exp['test_avg'],
                'best_test_win': best_exp['test_win_rate'],
                'experiments': model_experiments
            })
    
    # 保存结果
    output_file = os.path.join(MODEL_OUTPUT_DIR, 'ppo_optimize_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    
    for exp in all_experiments:
        print(f"\n{exp['model']}:")
        print(f"  最佳实验: {exp['best_exp']}")
        print(f"  Test收益: {exp['best_test_avg']*100:.2f}%")
        print(f"  Test胜率: {exp['best_test_win']:.1f}%")
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
