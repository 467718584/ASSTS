"""
PPO强化学习训练代码 - 优化版本
改进点：
1. 改进奖励函数设计
2. 调整超参数
3. 增加网络容量
4. 改进探索策略
5. 添加奖励归一化
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import random
import json
from datetime import datetime

# 添加项目路径
sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'
TRAINING_LOG_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_logs'

# 优化后的超参数
ACTOR_LR = 1e-4          # 降低学习率
CRITIC_LR = 5e-4
GAMMA = 0.99            # 折扣因子
LAMBDA = 0.95           # GAE参数
EPS_CLIP = 0.15         # PPO截断范围
K_EPOCHS = 10           # 增加更新轮数
UPDATE_INTERVAL = 256    # 增加更新间隔
MAX_EPISODES = 1000     # 增加训练轮数
WINDOW_SIZE = 10
HIDDEN_DIM = 128        # 增加隐藏层维度
ENTROPY_COEF = 0.01     # 添加熵系数促进探索
VALUE_COEF = 0.5        # 价值损失系数

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)


class PPOMemory:
    """PPO经验回放缓冲区"""
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
        # 奖励归一化
        rewards = np.array(self.rewards, dtype=np.float32)
        if len(rewards) > 1 and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            rewards,
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )


class Actor(nn.Module):
    """策略网络 - 改进版"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """价值网络 - 改进版"""
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """PPO智能体 - 优化版"""
    def __init__(self, state_dim, action_dim, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM)
        self.critic = Critic(state_dim, HIDDEN_DIM)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = GAMMA
        self.lambda_ = LAMBDA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        self.entropy_coef = ENTROPY_COEF
        self.value_coef = VALUE_COEF
        
        self.memory = PPOMemory()
        
        # 动作选择历史
        self.action_history = deque(maxlen=100)
    
    def select_action(self, state, training=True):
        """选择动作 - 带温度参数"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        probs = self.actor(state_tensor)
        value = self.critic(state_tensor)
        
        # 添加小的噪声防止概率为0
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.action_history.append(action.item())
        else:
            action = probs.argmax(dim=1)
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self):
        """更新网络 - 带熵正则化"""
        states, actions, rewards, dones, old_log_probs, old_values = self.memory.get()
        
        # 计算returns和advantages
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
        
        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为tensor
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # K轮更新
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(self.k_epochs):
            # 获取当前策略的概率
            probs = self.actor(states)
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # 熵 (促进探索)
            entropy = dist.entropy().mean()
            
            # 计算策略损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # 价值损失
            values = self.critic(states).squeeze()
            critic_loss = self.value_coef * nn.MSELoss()(values, returns)
            
            # 更新
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
        
        self.memory.clear()
        
        return total_actor_loss / self.k_epochs, total_critic_loss / self.k_epochs


def train():
    """训练PPO智能体 - 优化版"""
    print("=" * 60)
    print("PPO强化学习训练 - 优化版")
    print("=" * 60)
    print(f"超参数:")
    print(f"  Actor LR: {ACTOR_LR}, Critic LR: {CRITIC_LR}")
    print(f"  Gamma: {GAMMA}, Lambda: {LAMBDA}")
    print(f"  EPS Clip: {EPS_CLIP}, K Epochs: {K_EPOCHS}")
    print(f"  Entropy Coef: {ENTROPY_COEF}, Value Coef: {VALUE_COEF}")
    print("=" * 60)
    
    # 选择股票池
    pool_paths = [
        os.path.join(POOL_DIR, 'ppo_pool_30d_2s3h_min1'),
        os.path.join(POOL_DIR, 'ppo_pool_60d_2s3h_min1'),
    ]
    valid_pools = [p for p in pool_paths if os.path.exists(p) and len(os.listdir(p)) > 10]
    
    if not valid_pools:
        print("没有找到有效的股票池!")
        return
    
    pool_dir = valid_pools[0]
    print(f"使用股票池: {pool_dir}")
    
    # 创建环境
    env = StockTradingEnv(pool_dir, window_size=WINDOW_SIZE)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建PPO智能体
    agent = PPOAgent(state_dim, action_dim)
    
    # 训练统计
    episode_rewards = []
    episode_profits = []
    losses = []
    sell_counts = []
    hold_counts = []
    
    best_reward = -float('inf')
    best_profit = -float('inf')
    
    # 训练循环
    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        total_profit = 0
        done = False
        sell_count = 0
        hold_count = 0
        
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 统计动作
            if action == 1:
                sell_count += 1
            else:
                hold_count += 1
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.memory.add(state, action, reward, done, log_prob, value)
            
            # 周期性更新
            if len(agent.memory.states) >= UPDATE_INTERVAL:
                loss = agent.update()
                losses.append(loss)
            
            total_reward += reward
            if 'profit' in info:
                total_profit = info['profit']
            
            state = next_state
        
        # 记录统计
        episode_rewards.append(total_reward)
        episode_profits.append(total_profit)
        sell_counts.append(sell_count)
        hold_counts.append(hold_count)
        
        # 打印进度
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_profit = np.mean(episode_profits[-20:])
            avg_sell = np.mean(sell_counts[-20:])
            avg_hold = np.mean(hold_counts[-20:])
            print(f"Episode {episode+1}/{MAX_EPISODES} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Profit: {avg_profit*100:.2f}% | "
                  f"Sell/Hold: {avg_sell:.1f}/{avg_hold:.1f}")
        
        # 保存最佳模型 (基于收益)
        if total_profit > best_profit:
            best_profit = total_profit
            best_reward = total_reward
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, os.path.join(MODEL_OUTPUT_DIR, 'best_ppo_model.pth'))
            print(f"  -> 保存最佳模型, Profit: {best_profit*100:.2f}%")
    
    # 保存最终模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, os.path.join(MODEL_OUTPUT_DIR, 'ppo_model_final.pth'))
    
    # 保存训练日志
    log_data = {
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'sell_counts': sell_counts,
        'hold_counts': hold_counts,
        'losses': losses,
        'config': {
            'actor_lr': ACTOR_LR,
            'critic_lr': CRITIC_LR,
            'gamma': GAMMA,
            'lambda': LAMBDA,
            'eps_clip': EPS_CLIP,
            'k_epochs': K_EPOCHS,
            'update_interval': UPDATE_INTERVAL,
            'max_episodes': MAX_EPISODES,
            'window_size': WINDOW_SIZE,
            'hidden_dim': HIDDEN_DIM,
            'entropy_coef': ENTROPY_COEF,
            'value_coef': VALUE_COEF
        }
    }
    
    log_file = os.path.join(TRAINING_LOG_DIR, f'training_log_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n训练完成! 日志保存至: {log_file}")
    print(f"最佳收益: {best_profit*100:.2f}%")
    
    # 绘制训练曲线
    plot_training_curves(episode_rewards, episode_profits, sell_counts, hold_counts, losses)


def plot_training_curves(rewards, profits, sell_counts, hold_counts, losses):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 奖励曲线
    axes[0, 0].plot(rewards, alpha=0.4, label='Raw')
    if len(rewards) > 20:
        axes[0, 0].plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), 
                       label='MA-20', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 收益曲线
    axes[0, 1].plot([p*100 for p in profits], alpha=0.4, label='Raw')
    if len(profits) > 20:
        axes[0, 1].plot(np.convolve([p*100 for p in profits], np.ones(20)/20, mode='valid'), 
                       label='MA-20', linewidth=2)
    axes[0, 1].set_title('Episode Profits (%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Profit (%)')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 累计收益
    cumulative = np.cumsum(profits)
    axes[0, 2].plot(cumulative * 100)
    axes[0, 2].set_title('Cumulative Profit (%)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Cumulative Profit (%)')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 动作分布
    axes[1, 0].plot(sell_counts, alpha=0.5, label='Sell', color='red')
    axes[1, 0].plot(hold_counts, alpha=0.5, label='Hold', color='blue')
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 损失曲线
    if len(losses) > 0:
        actor_losses = [l[0] for l in losses]
        critic_losses = [l[1] for l in losses]
        axes[1, 1].plot(actor_losses, label='Actor Loss', alpha=0.7)
        axes[1, 1].plot(critic_losses, label='Critic Loss', alpha=0.7)
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 收益分布直方图
    axes[1, 2].hist([p*100 for p in profits], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Profit Distribution (%)')
    axes[1, 2].set_xlabel('Profit (%)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = os.path.join(TRAINING_LOG_DIR, 'training_curves_optimized.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {plot_file}")
    
    plt.close()


if __name__ == '__main__':
    train()
