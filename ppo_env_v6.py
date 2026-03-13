"""
PPO强化学习交易环境 - V6版本
基于波动率和持仓表现的奖励函数
"""
import os
import numpy as np
import pandas as pd
from collections import deque
import random

class StockTradingEnvV6:
    """
    V6版：更智能的奖励函数
    
    奖励设计：
    1. 基础收益奖励
    2. 持仓期间最大收益bonus (鼓励等到更高点)
    3. 波动率惩罚 (避免过早卖出)
    4. 晚卖bonus (收盘前卖出给更多奖励)
    """
    
    def __init__(self, stock_data_dir, stock_list=None, window_size=10, initial_capital=10000):
        self.stock_data_dir = stock_data_dir
        self.window_size = window_size
        self.initial_capital = initial_capital
        
        if stock_list:
            self.stock_codes = stock_list
        else:
            self.stock_codes = []
            for d in os.listdir(stock_data_dir):
                full_path = os.path.join(stock_data_dir, d)
                if os.path.isdir(full_path):
                    csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
                    if len(csv_files) >= 3:
                        self.stock_codes.append(d)
        
        if len(self.stock_codes) == 0:
            raise ValueError("No valid stocks")
        
        self.action_space = 2
        self.observation_space = window_size * 5
        self.stop_loss = -0.02
        self.transaction_fee = 0.0
        self.reset()
    
    def _load_stock_data(self, stock_code):
        stock_dir = os.path.join(self.stock_data_dir, stock_code)
        csv_files = sorted([f for f in os.listdir(stock_dir) if f.endswith('.csv')])
        if len(csv_files) < 3: return None
        day1 = pd.read_csv(os.path.join(stock_dir, csv_files[0]))
        day2 = pd.read_csv(os.path.join(stock_dir, csv_files[1]))
        day3 = pd.read_csv(os.path.join(stock_dir, csv_files[2]))
        return {'day1': day1, 'day2': day2, 'day3': day3}
    
    def _get_observation(self):
        self.profit_ratio = (self.current_price - self.buy_price) / self.buy_price if self.buy_price > 0 else 0
        self.drawdown = (self.peak_price - self.current_price) / self.peak_price if self.peak_price > 0 else 0
        self.profit_ratio = max(-1, min(1, self.profit_ratio)) if np.isfinite(self.profit_ratio) else 0
        self.drawdown = max(0, min(1, self.drawdown)) if np.isfinite(self.drawdown) else 0
        self.time_decay = self.day3_step / self.day3_max_steps if self.day3_max_steps > 0 else 0
        
        if len(self.price_history) >= 5:
            prices = list(self.price_history)[-5:]
            self.volatility = np.std(prices) / self.buy_price if self.buy_price > 0 else 0
        else:
            self.volatility = 0
        self.volatility = max(0, min(1, self.volatility)) if np.isfinite(self.volatility) else 0
        
        obs = np.array([self.profit_ratio, self.drawdown, self.time_decay, self.volatility, 
                       1.0 if self.position == 1 else 0.0], dtype=np.float32)
        
        self.obs_history.append(obs)
        if len(self.obs_history) > self.window_size: self.obs_history = self.obs_history[-self.window_size:]
        if len(self.obs_history) < self.window_size:
            padding = np.zeros((self.window_size - len(self.obs_history), 5), dtype=np.float32)
            obs_history = np.vstack([padding, np.array(self.obs_history, dtype=np.float32)])
        else:
            obs_history = np.array(self.obs_history[-self.window_size:], dtype=np.float32)
        return obs_history.flatten()
    
    def reset(self):
        self.current_stock = random.choice(self.stock_codes)
        self.data = self._load_stock_data(self.current_stock)
        if self.data is None or len(self.data['day2']) < 10 or len(self.data['day3']) < 10: return self.reset()
        
        self.buy_price = self.data['day2']['收盘'].iloc[0]
        self.cash = self.initial_capital
        self.position = 1
        self.current_price = self.buy_price
        self.peak_price = self.buy_price
        self.profit_ratio = self.drawdown = self.time_decay = self.volatility = 0
        
        # V6新增：追踪持仓期间最大收益
        self.max_profit_during_hold = 0
        
        self.day3_max_steps = min(len(self.data['day3']) - 1, 240)
        self.day3_step = 0
        self.price_history = deque(maxlen=60)
        self.obs_history = []
        self.done = False
        self.total_profit = 0
        
        self.day3_step = 15
        if self.day3_step < self.day3_max_steps:
            self.current_price = self.data['day3'].iloc[self.day3_step]['收盘']
            self.price_history.append(self.current_price)
            self.peak_price = max(self.peak_price, self.current_price)
        
        return self._get_observation()
    
    def step(self, action):
        if self.position == 0:
            self.position = 1
            self.buy_price = self.current_price
            self.peak_price = self.buy_price
            self.max_profit_during_hold = 0
            obs = self._get_observation()
            return obs, 0, self.done, {}
        
        self.day3_step += 1
        
        # 更新持仓期间最大收益
        current_profit = (self.current_price - self.buy_price) / self.buy_price
        self.max_profit_during_hold = max(self.max_profit_during_hold, current_profit)
        
        if self.day3_step >= self.day3_max_steps:
            # ============ V6奖励函数 ============
            # 基础收益
            base_reward = self.profit_ratio
            
            # 持仓期间最大收益bonus (如果卖在次高点，给额外奖励)
            max_bonus = (self.max_profit_during_hold - self.profit_ratio) * 0.5
            
            # 晚卖bonus (越晚卖，bonus越高)
            time_bonus = (self.day3_step / self.day3_max_steps) * 0.2
            
            reward = base_reward + max_bonus + time_bonus - self.transaction_fee
            
            self.position = 0
            self.done = True
            self.total_profit = reward
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock, 'sell_step': self.day3_step}
            return obs, reward, self.done, info
        
        self.current_price = self.data['day3'].iloc[self.day3_step]['收盘']
        self.price_history.append(self.current_price)
        self.peak_price = max(self.peak_price, self.current_price)
        
        # 止损
        if self.profit_ratio <= self.stop_loss:
            reward = self.profit_ratio * 0.5 - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock, 'stop_loss': True}
            return obs, reward, self.done, info
        
        if action == 1:  # 卖出
            # ============ V6奖励函数 ============
            # 基础收益
            base_reward = self.profit_ratio
            
            # 持仓期间最大收益bonus
            max_bonus = (self.max_profit_during_hold - self.profit_ratio) * 0.5
            
            # 晚卖bonus
            time_bonus = (self.day3_step / self.day3_max_steps) * 0.2
            
            reward = base_reward + max_bonus + time_bonus - self.transaction_fee
            
            self.position = 0
            self.done = True
            self.total_profit = reward
        else:
            # 持有：给小额负奖励，鼓励卖出
            reward = -0.0001
        
        obs = self._get_observation()
        info = {'profit': self.total_profit if self.done else 0}
        return obs, reward, self.done, info
