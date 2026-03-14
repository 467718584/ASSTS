"""
PPO强化学习交易环境 - 严格版
逻辑：
- Day1: 模型推荐股票
- Day2: 开盘买入
- Day3: 选择卖出时机

环境设计：
- 每只股票的分钟数据必须包含3天
- Day2的数据用于买入（开盘价）
- Day3的数据用于选择卖出时机
"""
import os
import numpy as np
import pandas as pd
from collections import deque
import random

class StockTradingEnv:
    """
    股票日内择时交易环境
    
    数据结构（3天）：
    - 文件[0] = Day1数据（推荐日，无用）
    - 文件[1] = Day2数据（买入日，开盘价买入）
    - 文件[2] = Day3数据（卖出日，选择卖出时机）
    
    状态: 比例特征
    动作: 0=持有, 1=卖出
    奖励: 卖出时结算真实盈亏
    """
    
    def __init__(self, stock_data_dir, window_size=10, initial_capital=10000):
        """
        :param stock_data_dir: 股票分钟数据目录（每只股票必须有3天数据）
        :param window_size: 状态窗口大小
        :param initial_capital: 初始资金
        """
        self.stock_data_dir = stock_data_dir
        self.window_size = window_size
        self.initial_capital = initial_capital
        
        # 获取所有股票（必须有3天数据）
        self.stock_codes = []
        for d in os.listdir(stock_data_dir):
            full_path = os.path.join(stock_data_dir, d)
            if os.path.isdir(full_path):
                csv_files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
                if len(csv_files) >= 3:  # 必须有3天
                    self.stock_codes.append(d)
        
        if len(self.stock_codes) == 0:
            raise ValueError(f"No valid stocks with 3 days data in {stock_data_dir}")
        
        print(f"    环境加载: {len(self.stock_codes)} 只股票（3天数据）")
        
        # 环境参数
        self.action_space = 2  # 0:持有, 1:卖出
        self.observation_space = window_size * 5
        
        # 交易参数
        self.stop_loss = -0.01
        self.transaction_fee = 0
        
        self.reset()
    
    def _load_stock_data(self, stock_code):
        """
        加载单只股票的3天分钟数据
        返回: [Day1数据, Day2数据, Day3数据]
        """
        stock_dir = os.path.join(self.stock_data_dir, stock_code)
        csv_files = sorted([f for f in os.listdir(stock_dir) if f.endswith('.csv')])
        
        if len(csv_files) < 3:
            return None
        
        day1_data = pd.read_csv(os.path.join(stock_dir, csv_files[0]))  # Day1 - 推荐日
        day2_data = pd.read_csv(os.path.join(stock_dir, csv_files[1]))  # Day2 - 买入日
        day3_data = pd.read_csv(os.path.join(stock_dir, csv_files[2]))  # Day3 - 卖出日
        
        return {
            'day1': day1_data,
            'day2': day2_data,  # 买入日
            'day3': day3_data,  # 卖出日
        }
    
    def _get_observation(self):
        """获取状态观测"""
        if self.current_price > 0 and self.buy_price > 0:
            self.profit_ratio = (self.current_price - self.buy_price) / self.buy_price
        else:
            self.profit_ratio = 0
            
        if self.peak_price > 0:
            self.drawdown = (self.peak_price - self.current_price) / self.peak_price
        else:
            self.drawdown = 0
        
        self.profit_ratio = max(-1, min(1, self.profit_ratio)) if np.isfinite(self.profit_ratio) else 0
        self.drawdown = max(0, min(1, self.drawdown)) if np.isfinite(self.drawdown) else 0
            
        self.time_decay = self.day3_step / self.day3_max_steps if self.day3_max_steps > 0 else 0
        
        if len(self.price_history) >= 5:
            prices = list(self.price_history)[-5:]
            self.volatility = np.std(prices) / self.buy_price if self.buy_price > 0 else 0
        else:
            self.volatility = 0
        
        self.volatility = max(0, min(1, self.volatility)) if np.isfinite(self.volatility) else 0
        
        obs = np.array([
            self.profit_ratio,
            self.drawdown,
            self.time_decay,
            self.volatility,
            1.0 if self.position == 1 else 0.0
        ], dtype=np.float32)
        
        self.obs_history.append(obs)
        
        if len(self.obs_history) > self.window_size:
            self.obs_history = self.obs_history[-self.window_size:]
        
        if len(self.obs_history) < self.window_size:
            padding = np.zeros((self.window_size - len(self.obs_history), 5), dtype=np.float32)
            obs_history = np.vstack([padding, np.array(self.obs_history, dtype=np.float32)])
        else:
            obs_history = np.array(self.obs_history[-self.window_size:], dtype=np.float32)
        
        return obs_history.flatten()
    
    def reset(self):
        """重置环境"""
        self.current_stock = random.choice(self.stock_codes)
        self.data = self._load_stock_data(self.current_stock)
        
        if self.data is None or len(self.data['day2']) < 10 or len(self.data['day3']) < 10:
            return self.reset()
        
        # ============ Day2: 开盘买入 ============
        day2_data = self.data['day2']
        
        # 开盘价买入（第一个价格）
        self.buy_price = day2_data['收盘'].iloc[0]  # 开盘价
        
        self.cash = self.initial_capital
        self.position = 1  # 持仓
        self.current_price = self.buy_price
        self.peak_price = self.buy_price
        self.profit_ratio = 0
        self.drawdown = 0
        self.time_decay = 0
        self.volatility = 0
        
        # Day3交易参数
        day3_data = self.data['day3']
        self.day3_max_steps = min(len(day3_data) - 1, 240)
        self.day3_step = 0
        
        self.price_history = deque(maxlen=60)
        self.obs_history = []
        self.done = False
        self.total_profit = 0
        
        # Day3跳过开盘前15分钟
        self.day3_step = 0
        
        if self.day3_step < self.day3_max_steps:
            self.current_price = day3_data.iloc[self.day3_step]['收盘']
            self.price_history.append(self.current_price)
            self.peak_price = max(self.peak_price, self.current_price)
        
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        """
        执行动作 - 在Day3选择卖出时机
        action: 0=持有, 1=卖出
        """
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
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock}
            return obs, reward, self.done, info
        
        day3_data = self.data['day3']
        self.current_price = day3_data.iloc[self.day3_step]['收盘']
        self.price_history.append(self.current_price)
        self.peak_price = max(self.peak_price, self.current_price)
        
        # 止损检查
        if self.profit_ratio <= self.stop_loss:
            reward = self.profit_ratio - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
            obs = self._get_observation()
            info = {'profit': reward, 'stock': self.current_stock, 'stop_loss': True}
            return obs, reward, self.done, info
        
        if action == 1:  # 卖出
            reward = self.profit_ratio - self.transaction_fee
            self.position = 0
            self.done = True
            self.total_profit = reward
        else:  # 持有
            reward = -0.0001
        
        obs = self._get_observation()
        info = {'profit': self.total_profit if self.done else 0}
        
        return obs, reward, self.done, info
