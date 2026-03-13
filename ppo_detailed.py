"""
PPO训练 - 详细结果输出
输出每只股票的买入卖出详情
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
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'

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
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class StockTradingEnvDetailed(StockTradingEnv):
    """详细环境 - 记录每笔交易的详细信息"""
    
    def __init__(self, stock_data_dir, stock_list, window_size=10, initial_capital=10000):
        self.stock_data_dir = stock_data_dir
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.stock_codes = stock_list
        
        self.action_space = 2
        self.observation_space = window_size * 5
        self.stop_loss = -0.03
        self.transaction_fee = 0.001
        
        self.reset()
    
    def reset(self):
        self.current_stock = random.choice(self.stock_codes)
        self.data = self._load_stock_data(self.current_stock)
        
        if self.data is None or len(self.data['day2']) < 10 or len(self.data['day3']) < 10:
            return self.reset()
        
        # 记录交易详情
        self.trade_details = {
            'stock_code': self.current_stock,
            'buy_price': None,
            'buy_time': None,
            'sell_price': None,
            'sell_time': None,
            'profit': None,
            'action_history': []
        }
        
        day2_data = self.data['day2']
        day3_data = self.data['day3']
        
        # 开盘价买入
        self.buy_price = day2_data['收盘'].iloc[0]
        self.trade_details['buy_price'] = float(self.buy_price)
        self.trade_details['buy_time'] = f"Day2 00:00"
        
        self.cash = self.initial_capital
        self.position = 1
        self.current_price = self.buy_price
        self.peak_price = self.buy_price
        self.profit_ratio = 0
        self.drawdown = 0
        self.time_decay = 0
        self.volatility = 0
        
        self.day3_max_steps = min(len(day3_data) - 1, 240)
        self.day3_step = 0
        
        self.price_history = deque(maxlen=60)
        self.obs_history = []
        self.done = False
        self.total_profit = 0
        
        self.day3_step = 15
        
        if self.day3_step < self.day3_max_steps:
            self.current_price = day3_data.iloc[self.day3_step]['收盘']
            self.price_history.append(self.current_price)
            self.peak_price = max(self.peak_price, self.current_price)
        
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        if self.position == 0:
            self.position = 1
            self.buy_price = self.current_price
            self.peak_price = self.buy_price
            obs = self._get_observation()
            reward = 0
            return obs, reward, self.done, {}
        
        self.day3_step += 1
        
        if self.day3_step >= self.day3_max_steps:
            reward = self.profit_ratio - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
            self.trade_details['sell_price'] = float(self.current_price)
            self.trade_details['sell_time'] = f"Day3 {self.day3_step}"
            self.trade_details['profit'] = float(reward)
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock, 'details': self.trade_details}
            return obs, reward, self.done, info
        
        day3_data = self.data['day3']
        self.current_price = day3_data.iloc[self.day3_step]['收盘']
        self.price_history.append(self.current_price)
        self.peak_price = max(self.peak_price, self.current_price)
        
        # 记录动作
        self.trade_details['action_history'].append({
            'step': self.day3_step,
            'action': 'sell' if action == 1 else 'hold',
            'price': float(self.current_price),
            'profit': float(self.profit_ratio)
        })
        
        if self.profit_ratio <= self.stop_loss:
            reward = self.profit_ratio - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
            self.trade_details['sell_price'] = float(self.current_price)
            self.trade_details['sell_time'] = f"Day3 {self.day3_step} (止损)"
            self.trade_details['profit'] = float(reward)
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock, 'stop_loss': True, 'details': self.trade_details}
            return obs, reward, self.done, info
        
        if action == 1:
            reward = self.profit_ratio - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
            self.trade_details['sell_price'] = float(self.current_price)
            self.trade_details['sell_time'] = f"Day3 {self.day3_step}"
            self.trade_details['profit'] = float(reward)
        else:
            reward = -0.0001
        
        obs = self._get_observation()
        info = {'profit': self.total_profit if self.done else 0, 'details': self.trade_details}
        
        return obs, reward, self.done, info


def split_dataset(stock_codes, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    random.shuffle(stock_codes)
    n = len(stock_codes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_stocks = stock_codes[:n_train]
    val_stocks = stock_codes[n_train:n_train+n_val]
    test_stocks = stock_codes[n_train+n_val:]
    
    return train_stocks, val_stocks, test_stocks


def train_and_evaluate(pool_path, model_name):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 获取股票
    all_stocks = [d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))]
    all_stocks = [s for s in all_stocks if len([f for f in os.listdir(os.path.join(pool_path, s)) if f.endswith('.csv')]) >= 3]
    
    train_stocks, val_stocks, test_stocks = split_dataset(all_stocks)
    
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"数据集: 训练={len(train_stocks)}, 验证={len(val_stocks)}, 测试={len(test_stocks)}")
    print(f"{'='*60}")
    
    # 训练
    env = StockTradingEnvDetailed(pool_path, train_stocks, window_size=WINDOW_SIZE)
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
        
        if episode % 20 == 0 and len(val_stocks) > 0:
            val_env = StockTradingEnvDetailed(pool_path, val_stocks, window_size=WINDOW_SIZE)
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
    
    if best_agent is None:
        best_agent = agent
    
    # 详细评估
    def evaluate_set(stock_list, set_name):
        if len(stock_list) == 0:
            return [], []
        
        eval_env = StockTradingEnvDetailed(pool_path, stock_list, window_size=WINDOW_SIZE)
        results = []
        
        for _ in range(len(stock_list)):
            state = eval_env.reset()
            done = False
            while not done:
                action, _, _ = best_agent.select_action(state, training=False)
                state, reward, done, info = eval_env.step(action)
            
            if 'details' in info:
                results.append(info['details'])
        
        profits = [r['profit'] for r in results if r.get('profit') is not None]
        
        print(f"\n--- {set_name} 详细结果 ---")
        print(f"样本数: {len(results)}")
        print(f"盈利: {len([p for p in profits if p > 0])} ({100*len([p for p in profits if p > 0])/len(profits):.1f}%)")
        print(f"亏损: {len([p for p in profits if p < 0])} ({100*len([p for p in profits if p < 0])/len(profits):.1f}%)")
        print(f"最佳收益: {max(profits)*100:.2f}%")
        print(f"平均收益: {np.mean(profits)*100:.2f}%")
        
        # 打印每只股票详情
        print(f"\n股票交易详情:")
        print(f"| 股票代码 | 买入价 | 卖出价 | 卖出时间 | 收益 |")
        print(f"|:-------:|------:|------:|:-------:|-----:|")
        
        for r in results[:20]:  # 只显示前20个
            code = r['stock_code']
            buy = r.get('buy_price', 0)
            sell = r.get('sell_price', 0)
            sell_time = r.get('sell_time', 'N/A')
            profit = r.get('profit', 0)
            print(f"| {code} | {buy:.4f} | {sell:.4f} | {sell_time} | {profit*100:.2f}% |")
        
        return results, profits
    
    train_results, train_profits = evaluate_set(train_stocks, "训练集")
    val_results, val_profits = evaluate_set(val_stocks, "验证集")
    test_results, test_profits = evaluate_set(test_stocks, "测试集")
    
    return {
        'model': model_name,
        'train': {'count': len(train_stocks), 'results': train_results},
        'val': {'count': len(val_stocks), 'results': val_results},
        'test': {'count': len(test_stocks), 'results': test_results},
    }


def main():
    print("="*60)
    print("PPO详细结果分析")
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
        
        if stock_count < 10:
            continue
        
        result = train_and_evaluate(pool_path, model_name)
        all_results.append(result)
        
        # 保存
        output_file = os.path.join(MODEL_OUTPUT_DIR, f'details_{model_name}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    
    for r in all_results:
        print(f"\n{r['model']}:")
        print(f"  训练集: {r['train']['count']} 只, 正收益率: {100*len([p for p in r['train']['results'] if p.get('profit',0)>0])/max(1,len(r['train']['results'])):.1f}%")
        print(f"  验证集: {r['val']['count']} 只, 正收益率: {100*len([p for p in r['val']['results'] if p.get('profit',0)>0])/max(1,len(r['val']['results'])):.1f}%")
        print(f"  测试集: {r['test']['count']} 只, 正收益率: {100*len([p for p in r['test']['results'] if p.get('profit',0)>0])/max(1,len(r['test']['results'])):.1f}%")


if __name__ == '__main__':
    main()
