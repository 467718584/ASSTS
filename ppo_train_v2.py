"""
PPO强化学习训练代码 - 深度优化版 v2
改进点：
1. 使用多个股票池增加样本多样性
2. 改进奖励函数 (添加shape reward)
3. 课程学习策略
4. 更长的训练周期
5. 学习率调度
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

sys.path.append('/home/zzy/project/ASSTS/ASSTS-stock_class')
from ppo_env import StockTradingEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'
TRAINING_LOG_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_logs'

# 深度优化超参数
ACTOR_LR = 3e-5          # 更低的学习率
CRITIC_LR = 1e-4
GAMMA = 0.995           # 更高的折扣因子
LAMBDA = 0.98
EPS_CLIP = 0.1          # 更小的截断范围
K_EPOCHS = 20           # 更多更新轮数
UPDATE_INTERVAL = 512   # 更大更新间隔
MAX_EPISODES = 2000     # 更长训练周期
WINDOW_SIZE = 10
HIDDEN_DIM = 256        # 更大隐藏层
ENTROPY_COEF = 0.02     # 更大熵系数促进探索
VALUE_COEF = 1.0
LR_DECAY = 0.995        # 学习率衰减
MIN_LR = 1e-6

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
        # 奖励归一化 + 奖励塑形
        rewards = np.array(self.rewards, dtype=np.float32)
        
        # 添加奖励塑形 - 鼓励早期卖出获得正反馈
        shaped_rewards = []
        for r in rewards:
            if r > 0:
                shaped_rewards.append(r * 2)  # 放大正奖励
            elif r < -0.02:
                shaped_rewards.append(r * 0.5)  # 缩小过大负奖励
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
    """深度策略网络"""
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
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """深度价值网络"""
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
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """深度PPO智能体"""
    def __init__(self, state_dim, action_dim, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM)
        self.critic = Critic(state_dim, HIDDEN_DIM)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.scheduler_actor = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=LR_DECAY)
        self.scheduler_critic = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=LR_DECAY)
        
        self.gamma = GAMMA
        self.lambda_ = LAMBDA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        self.entropy_coef = ENTROPY_COEF
        self.value_coef = VALUE_COEF
        self.min_lr = MIN_LR
        
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
        
        # 学习率调度
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        
        self.memory.clear()
        
        return total_actor_loss / self.k_epochs, total_critic_loss / self.k_epochs
    
    def get_lr(self):
        return self.actor_optimizer.param_groups[0]['lr']


def train():
    print("=" * 60)
    print("PPO强化学习训练 - 深度优化版 v2")
    print("=" * 60)
    print(f"超参数:")
    print(f"  Actor LR: {ACTOR_LR}, Critic LR: {CRITIC_LR}")
    print(f"  Gamma: {GAMMA}, Lambda: {LAMBDA}")
    print(f"  EPS Clip: {EPS_CLIP}, K Epochs: {K_EPOCHS}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Max Episodes: {MAX_EPISODES}")
    print("=" * 60)
    
    # 使用多个股票池
    pool_paths = [
        os.path.join(POOL_DIR, 'ppo_pool_30d_2s3h_min1'),
        os.path.join(POOL_DIR, 'ppo_pool_60d_2s3h_min1'),
        os.path.join(POOL_DIR, 'ppo_pool_combined_min1'),
    ]
    valid_pools = [p for p in pool_paths if os.path.exists(p) and len(os.listdir(p)) > 10]
    
    if not valid_pools:
        print("没有找到有效的股票池!")
        return
    
    print(f"使用 {len(valid_pools)} 个股票池")
    
    # 创建环境 - 轮换使用不同股票池
    pool_idx = 0
    env = StockTradingEnv(valid_pools[pool_idx], window_size=WINDOW_SIZE)
    state_dim = env.observation_space
    action_dim = env.action_space
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    agent = PPOAgent(state_dim, action_dim)
    
    episode_rewards = []
    episode_profits = []
    losses = []
    sell_counts = []
    hold_counts = []
    lr_history = []
    
    best_profit = -float('inf')
    patience = 0
    max_patience = 100
    
    for episode in range(MAX_EPISODES):
        # 课程学习 - 每200轮切换股票池
        if episode > 0 and episode % 200 == 0:
            pool_idx = (pool_idx + 1) % len(valid_pools)
            env = StockTradingEnv(valid_pools[pool_idx], window_size=WINDOW_SIZE)
            print(f"  -> 切换到股票池 {pool_idx+1}/{len(valid_pools)}")
        
        state = env.reset()
        total_reward = 0
        total_profit = 0
        done = False
        sell_count = 0
        hold_count = 0
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            
            if action == 1:
                sell_count += 1
            else:
                hold_count += 1
            
            next_state, reward, done, info = env.step(action)
            
            agent.memory.add(state, action, reward, done, log_prob, value)
            
            if len(agent.memory.states) >= UPDATE_INTERVAL:
                loss = agent.update()
                losses.append(loss)
            
            total_reward += reward
            if 'profit' in info:
                total_profit = info['profit']
            
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_profits.append(total_profit)
        sell_counts.append(sell_count)
        hold_counts.append(hold_count)
        lr_history.append(agent.get_lr())
        
        if (episode + 1) % 50 == 0:
            avg_profit = np.mean(episode_profits[-50:])
            avg_sell = np.mean(sell_counts[-50:])
            avg_hold = np.mean(hold_counts[-50:])
            current_lr = agent.get_lr()
            print(f"Episode {episode+1}/{MAX_EPISODES} | "
                  f"Avg Profit: {avg_profit*100:.2f}% | "
                  f"S/H: {avg_sell:.1f}/{avg_hold:.1f} | "
                  f"LR: {current_lr:.2e}")
        
        if total_profit > best_profit:
            best_profit = total_profit
            patience = 0
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, os.path.join(MODEL_OUTPUT_DIR, 'best_ppo_model.pth'))
            print(f"  -> NEW BEST! Profit: {best_profit*100:.2f}%")
        else:
            patience += 1
        
        # 早停
        if patience > max_patience and episode > 500:
            print(f"早停触发! 连续{max_patience}轮未创新高")
            break
    
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, os.path.join(MODEL_OUTPUT_DIR, 'ppo_model_final.pth'))
    
    log_data = {
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
        'sell_counts': sell_counts,
        'hold_counts': hold_counts,
        'lr_history': lr_history,
        'losses': losses,
        'best_profit': best_profit,
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
            'value_coef': VALUE_COEF,
            'lr_decay': LR_DECAY
        }
    }
    
    log_file = os.path.join(TRAINING_LOG_DIR, f'training_log_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n训练完成! 最佳收益: {best_profit*100:.2f}%")
    print(f"日志保存至: {log_file}")
    
    plot_training_curves(episode_rewards, episode_profits, sell_counts, hold_counts, losses, lr_history)


def plot_training_curves(rewards, profits, sell_counts, hold_counts, losses, lr_history):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    
    # 奖励
    axes[0, 0].plot(rewards, alpha=0.3)
    if len(rewards) > 50:
        axes[0, 0].plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 收益
    axes[0, 1].plot([p*100 for p in profits], alpha=0.3)
    if len(profits) > 50:
        axes[0, 1].plot(np.convolve([p*100 for p in profits], np.ones(50)/50, mode='valid'), linewidth=2)
    axes[0, 1].set_title('Episode Profits (%)')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 累计收益
    cumulative = np.cumsum(profits)
    axes[0, 2].plot(cumulative * 100)
    axes[0, 2].set_title('Cumulative Profit (%)')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 学习率
    axes[0, 3].plot(lr_history)
    axes[0, 3].set_title('Learning Rate')
    axes[0, 3].set_yscale('log')
    axes[0, 3].grid(True, alpha=0.3)
    
    # 动作分布
    axes[1, 0].plot(sell_counts, alpha=0.5, label='Sell', color='red')
    axes[1, 0].plot(hold_counts, alpha=0.5, label='Hold', color='blue')
    axes[1, 0].set_title('Action Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 损失
    if len(losses) > 0:
        actor_losses = [l[0] for l in losses]
        critic_losses = [l[1] for l in losses]
        axes[1, 1].plot(actor_losses, label='Actor', alpha=0.7)
        axes[1, 1].plot(critic_losses, label='Critic', alpha=0.7)
        axes[1, 1].set_title('Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 收益分布
    axes[1, 2].hist([p*100 for p in profits], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Profit Distribution (%)')
    axes[1, 2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 2].grid(True, alpha=0.3)
    
    # 滑动平均收益
    window = 50
    ma_profits = []
    for i in range(len(profits)):
        start = max(0, i - window + 1)
        ma_profits.append(np.mean(profits[start:i+1]) * 100)
    axes[1, 3].plot(ma_profits)
    axes[1, 3].set_title(f'Moving Avg Profit (window={window})')
    axes[1, 3].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(TRAINING_LOG_DIR, 'training_curves_v2.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {plot_file}")
    plt.close()


if __name__ == '__main__':
    train()
