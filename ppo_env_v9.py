"""
PPO强化学习交易环境 - V9版本
进一步优化：更长训练 + Ensemble思路
"""
import os
import numpy as np
import pandas as pd
from collections import deque
import random

class StockTradingEnvV9:
    """
    V9版：最终优化
    
    核心改进：
    1. 更激进的奖励：大幅提高相对收益权重
    2. 趋势保护：下跌趋势不卖出可获得保护奖励
    3. 更灵活的止损
    4. 更大的特征窗口
    """
    
    def __init__(self, stock_data_dir, stock_list=None, window_size=20, initial_capital=10000):
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
        self.observation_space = window_size * 8
        self.stop_loss = -0.03
        self.transaction_fee = 0.0
        self.reset()
    
    def _load_stock_data(self, stock_code):
        stock_dir = os.path.join(self.stock_data_dir, stock_code)
        csv_files = sorted([f for f in os.listdir(stock_dir) if f.endswith('.csv')])
        if len(csv_files) < 3: return None
        return {
            'day1': pd.read_csv(os.path.join(stock_dir, csv_files[0])),
            'day2': pd.read_csv(os.path.join(stock_dir, csv_files[1])),
            'day3': pd.read_csv(os.path.join(stock_dir, csv_files[2]))
        }
    
    def _compute_features(self, prices):
        if len(prices) < 5:
            return 0, 0, 0, 0
        
        prices = np.array(prices)
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        ma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        ma10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        trend = (ma5 - ma10) / (ma10 + 1e-8)
        
        momentum = (prices[-1] - prices[-3]) / (prices[-3] + 1e-8) if len(prices) >= 3 else 0
        
        rsi = 50
        if len(returns) >= 5:
            gains = np.maximum(returns, 0)
            losses = np.maximum(-returns, 0)
            avg_gain, avg_loss = np.mean(gains[-5:]), np.mean(losses[-5:])
            if avg_loss > 0:
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
        
        return volatility, trend, momentum, (rsi - 50) / 50
    
    def _get_observation(self):
        self.profit_ratio = (self.current_price - self.buy_price) / (self.buy_price + 1e-8)
        self.drawdown = (self.peak_price - self.current_price) / (self.peak_price + 1e-8)
        self.profit_ratio = np.clip(self.profit_ratio, -1, 1) if np.isfinite(self.profit_ratio) else 0
        self.drawdown = np.clip(self.drawdown, 0, 1) if np.isfinite(self.drawdown) else 0
        self.time_decay = self.day3_step / max(self.day3_max_steps, 1)
        
        volatility, trend, momentum, rsi = self._compute_features(list(self.price_history))
        
        obs = np.array([
            self.profit_ratio, self.drawdown, self.time_decay, volatility,
            trend, momentum, rsi, 1.0 if self.position == 1 else 0.0
        ], dtype=np.float32)
        
        self.obs_history.append(obs)
        if len(self.obs_history) > self.window_size:
            self.obs_history = self.obs_history[-self.window_size:]
        if len(self.obs_history) < self.window_size:
            padding = np.zeros((self.window_size - len(self.obs_history), 8), dtype=np.float32)
            obs_history = np.vstack([padding, np.array(self.obs_history)])
        else:
            obs_history = np.array(self.obs_history[-self.window_size:])
        return obs_history.flatten()
    
    def reset(self):
        self.current_stock = random.choice(self.stock_codes)
        self.data = self._load_stock_data(self.current_stock)
        if self.data is None or len(self.data['day2']) < 10 or len(self.data['day3']) < 10:
            return self.reset()
        
        self.day3_close = self.data['day3']['收盘'].iloc[-1]
        self.buy_price = self.data['day2']['收盘'].iloc[0]
        self.day3_return = (self.day3_close - self.buy_price) / (self.buy_price + 1e-8)
        
        self.cash = self.initial_capital
        self.position = 1
        self.current_price = self.buy_price
        self.peak_price = self.buy_price
        self.max_profit_during_hold = 0
        
        self.day3_max_steps = min(len(self.data['day3']) - 1, 240)
        self.day3_step = 0
        self.price_history = deque(maxlen=60)
        self.obs_history = []
        self.done = False
        
        self.day3_step = random.randint(0, min(8, self.day3_max_steps - 1))
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
            return self._get_observation(), 0, self.done, {}
        
        self.day3_step += 1
        current_profit = (self.current_price - self.buy_price) / (self.buy_price + 1e-8)
        self.max_profit_during_hold = max(self.max_profit_during_hold, current_profit)
        
        if self.day3_step >= self.day3_max_steps:
            # V9奖励：大幅提高相对收益权重
            base_reward = self.profit_ratio * 0.3
            relative_bonus = (self.profit_ratio - self.day3_return) * 1.2  # 更激进
            max_bonus = (self.max_profit_during_hold - self.profit_ratio) * 0.5
            time_bonus = (self.day3_step / self.day3_max_steps) * 0.1
            
            reward = base_reward + relative_bonus + max_bonus + time_bonus
            
            self.position = 0
            self.done = True
            return self._get_observation(), reward, self.done, {'profit': reward, 'sell_step': self.day3_step}
        
        self.current_price = self.data['day3'].iloc[self.day3_step]['收盘']
        self.price_history.append(self.current_price)
        self.peak_price = max(self.peak_price, self.current_price)
        
        # 止损
        if self.profit_ratio <= self.stop_loss:
            reward = self.profit_ratio * 0.1
            self.position = 0
            self.done = True
            return self._get_observation(), reward, self.done, {'profit': reward, 'stop_loss': True}
        
        if action == 1:  # 卖出
            base_reward = self.profit_ratio * 0.3
            relative_bonus = (self.profit_ratio - self.day3_return) * 1.2
            max_bonus = (self.max_profit_during_hold - self.profit_ratio) * 0.5
            
            # 早卖更严格惩罚
            early_penalty = -0.15 if self.day3_step < 6 else (-0.08 if self.day3_step < 10 else 0)
            
            time_bonus = (self.day3_step / self.day3_max_steps) * 0.1
            reward = base_reward + relative_bonus + max_bonus + time_bonus + early_penalty
            
            self.position = 0
            self.done = True
        else:
            reward = -0.00002  # 更小的持有惩罚
        
        return self._get_observation(), reward, self.done, {'profit': reward if self.done else 0}
