"""
PPO强化学习交易环境
专门用于学习最佳卖出时机
"""
import os
import numpy as np
import pandas as pd
from collections import deque
import random

class StockTradingEnv:
    """
    股票日内择时交易环境
    状态: 纯比例特征 (浮盈比例, 回撤比例, 时间衰减比例)
    动作: 0=持有, 1=卖出
    奖励: 卖出时结算真实盈亏
    """
    
    def __init__(self, stock_data_dir, window_size=10, initial_capital=10000):
        """
        :param stock_data_dir: 股票分钟数据目录
        :param window_size: 状态窗口大小
        :param initial_capital: 初始资金
        """
        self.stock_data_dir = stock_data_dir
        self.window_size = window_size
        self.initial_capital = initial_capital
        
        # 获取所有股票
        self.stock_codes = [d for d in os.listdir(stock_data_dir) 
                           if os.path.isdir(os.path.join(stock_data_dir, d))]
        
        # 环境参数
        self.action_space = 2  # 0:持有, 1:卖出
        self.observation_space = window_size * 5  # 5个特征 * window (profit, drawdown, time_decay, volatility, position)
        
        # 交易参数
        self.stop_loss = -0.03  # -3%止损
        self.transaction_fee = 0.001  # 手续费
        
        # 状态归一化参数
        self.max_drawdown = 0.1  # 最大回撤10%
        self.max_time_decay = 1.0  # 最大时间衰减
        
        self.reset()
    
    def _load_stock_data(self, stock_code):
        """加载单只股票的分钟数据"""
        stock_dir = os.path.join(self.stock_data_dir, stock_code)
        all_data = []
        
        for csv_file in sorted(os.listdir(stock_dir)):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(stock_dir, csv_file))
                if len(df) > 0:
                    all_data.append(df)
        
        if not all_data:
            return None
            
        data = pd.concat(all_data, ignore_index=True)
        return data
    
    def _get_observation(self):
        """
        获取状态观测
        状态包含:
        - 当前浮盈比例 (profit_ratio)
        - 距离最高点回撤 (drawdown_from_peak)
        - 时间衰减比例 (time_decay)
        - 实时波动率 (volatility)
        - 当前是否持仓 (position)
        """
        # 计算浮盈比例
        if self.current_price > 0 and self.buy_price > 0:
            self.profit_ratio = (self.current_price - self.buy_price) / self.buy_price
        else:
            self.profit_ratio = 0
            
        # 计算回撤
        if self.peak_price > 0:
            self.drawdown = (self.peak_price - self.current_price) / self.peak_price
        else:
            self.drawdown = 0
        
        # 防止NaN和Inf
        self.profit_ratio = max(-1, min(1, self.profit_ratio)) if np.isfinite(self.profit_ratio) else 0
        self.drawdown = max(0, min(1, self.drawdown)) if np.isfinite(self.drawdown) else 0
            
        # 时间衰减 (已交易时间 / 总交易时间)
        self.time_decay = self.current_step / self.max_steps if self.max_steps > 0 else 0
        
        # 波动率 (最近N个bar的价格波动)
        if len(self.price_history) >= 5:
            prices = list(self.price_history)[-5:] if hasattr(self.price_history, '__iter__') else self.price_history[-5:]
            self.volatility = np.std(prices) / self.buy_price if self.buy_price > 0 else 0
        else:
            self.volatility = 0
        
        # 防止NaN和Inf
        self.volatility = max(0, min(1, self.volatility)) if np.isfinite(self.volatility) else 0
        
        # 构建状态向量
        obs = np.array([
            self.profit_ratio,
            self.drawdown,
            self.time_decay,
            self.volatility,
            1.0 if self.position == 1 else 0.0  # 当前持仓状态
        ], dtype=np.float32)
        
        # 添加到历史
        self.obs_history.append(obs)
        
        # 确保历史记录只保留window_size个
        if len(self.obs_history) > self.window_size:
            self.obs_history = self.obs_history[-self.window_size:]
        
        # 填充窗口 - 确保固定输出大小
        if len(self.obs_history) < self.window_size:
            # 前面填充0
            padding = np.zeros((self.window_size - len(self.obs_history), 5), dtype=np.float32)
            obs_history = np.vstack([padding, np.array(self.obs_history, dtype=np.float32)])
        else:
            obs_history = np.array(self.obs_history[-self.window_size:], dtype=np.float32)
        
        return obs_history.flatten()
    
    def reset(self):
        """重置环境"""
        # 随机选择一只股票
        self.current_stock = random.choice(self.stock_codes)
        self.data = self._load_stock_data(self.current_stock)
        
        if self.data is None or len(self.data) < 60:
            return self.reset()  # 重新选择
        
        # 初始化交易状态
        self.cash = self.initial_capital
        self.position = 0  # 0:空仓, 1:持仓
        self.buy_price = 0
        self.current_price = 0
        self.peak_price = 0
        self.profit_ratio = 0
        self.drawdown = 0
        self.time_decay = 0
        self.volatility = 0
        
        # 时间步
        self.current_step = 0
        self.max_steps = min(len(self.data) - 1, 240)  # 最多4小时
        
        # 历史记录
        self.price_history = deque(maxlen=60)
        self.obs_history = []
        self.done = False
        self.total_profit = 0
        
        # 跳过开盘前30分钟 (9:30-10:00不稳定)
        self.current_step = 30
        
        if self.current_step < self.max_steps:
            self.current_price = self.data.iloc[self.current_step]['收盘']
            self.price_history.append(self.current_price)
            self.peak_price = self.current_price
            
            # 随机决定是否开盘买入
            if random.random() > 0.3:
                self.position = 1
                self.buy_price = self.current_price
        
        obs = self._get_observation()
        return obs
    
    def step(self, action):
        """
        执行动作
        action: 0=持有, 1=卖出
        """
        # 如果空仓，随机买入
        if self.position == 0:
            # 买入
            self.position = 1
            self.buy_price = self.current_price
            self.peak_price = self.current_price
            obs = self._get_observation()
            reward = 0
            return obs, reward, self.done, {}
        
        # 记录之前的状态
        prev_profit = self.profit_ratio
        
        # 更新价格
        self.current_step += 1
        if self.current_step >= self.max_steps:
            # 强制平仓
            if self.position == 1:
                reward = self.profit_ratio
                self.position = 0
            else:
                reward = 0
            self.done = True
            obs = self._get_observation()
            return obs, reward, self.done, {}
        
        # 获取当前价格
        self.current_price = self.data.iloc[self.current_step]['收盘']
        self.price_history.append(self.current_price)
        
        # 更新最高价
        if self.current_price > self.peak_price:
            self.peak_price = self.current_price
        
        reward = 0
        info = {}
        
        if action == 1 and self.position == 1:
            # 卖出
            profit = (self.current_price - self.buy_price) / self.buy_price
            profit -= self.transaction_fee  # 扣除手续费
            
            # 奖励 = 真实盈亏
            reward = profit
            self.total_profit = profit
            
            self.position = 0
            self.done = True
            info = {
                'profit': profit,
                'buy_price': self.buy_price,
                'sell_price': self.current_price,
                'stock': self.current_stock,
                'step': self.current_step
            }
        
        # 强制止损
        if self.position == 1 and self.profit_ratio < self.stop_loss:
            profit = self.profit_ratio - self.transaction_fee
            reward = profit - 0.02  # 额外惩罚
            self.position = 0
            self.done = True
            info = {
                'profit': profit,
                'buy_price': self.buy_price,
                'sell_price': self.current_price,
                'stock': self.current_stock,
                'step': self.current_step,
                'stop_loss': True
            }
        
        # 持仓时间惩罚 (防止死扛)
        if self.position == 1 and not self.done:
            reward -= 0.0001  # 每个时间步轻微惩罚
        
        obs = self._get_observation()
        
        # 保存观测历史
        if self.position == 1:
            self.obs_history.append(np.array([
                self.profit_ratio,
                self.drawdown,
                self.time_decay,
                self.volatility,
                1.0
            ]))
        
        return obs, reward, self.done, info
    
    def render(self, mode='human'):
        """渲染环境"""
        print(f"Stock: {self.current_stock}, Step: {self.current_step}/{self.max_steps}")
        print(f"Position: {self.position}, Price: {self.current_price:.2f}")
        print(f"Profit: {self.profit_ratio*100:.2f}%, Drawdown: {self.drawdown*100:.2f}%")
        print(f"Time Decay: {self.time_decay*100:.1f}%")
        print("-" * 40)


class MultiStockEnv:
    """多股票并行环境 - 加速训练"""
    
    def __init__(self, pool_dirs, window_size=10):
        """
        :param pool_dirs: 股票池目录列表
        """
        self.envs = [StockTradingEnv(d, window_size) for d in pool_dirs]
        self.n_envs = len(self.envs)
        
    def reset(self):
        return np.array([env.reset() for env in self.envs])
    
    def step(self, actions):
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            
            if done:
                env.reset()
        
        return np.array(obs_list), np.array(reward_list), np.array(done_list), info_list
