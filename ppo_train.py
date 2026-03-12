"""
PPO强化学习训练代码
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
from ppo_env import StockTradingEnv, MultiStockEnv

# 配置
POOL_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_pool'
MODEL_OUTPUT_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_model'
TRAINING_LOG_DIR = '/home/zzy/project/ASSTS/ASSTS-stock_class/ppo_logs'

# 超参数
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_INTERVAL = 128  # 每次更新收集的样本数
MAX_EPISODES = 500
WINDOW_SIZE = 10

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
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )


class Actor(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """价值网络"""
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """PPO智能体"""
    def __init__(self, state_dim, action_dim, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = GAMMA
        self.lambda_ = LAMBDA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        
        self.memory = PPOMemory()
    
    def select_action(self, state, training=True):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        dist = Categorical(probs)
        
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = probs.argmax(dim=1)
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self):
        """更新网络"""
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
        for _ in range(self.k_epochs):
            # 获取当前策略的概率
            probs = self.actor(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算策略损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)
            
            # 更新
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item()


def train():
    """训练PPO智能体"""
    print("=" * 60)
    print("PPO强化学习训练")
    print("=" * 60)
    
    # 选择股票池
    pool_paths = [
        os.path.join(POOL_DIR, 'ppo_pool_30d_2s3h_min1'),
        os.path.join(POOL_DIR, 'ppo_pool_60d_2s3h_min1'),
    ]
    # 只使用第一个有数据的池
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
    
    best_reward = -float('inf')
    
    # 训练循环
    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        total_profit = 0
        done = False
        
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
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
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_profit = np.mean(episode_profits[-10:])
            print(f"Episode {episode+1}/{MAX_EPISODES} | Avg Reward: {avg_reward:.4f} | Avg Profit: {avg_profit*100:.2f}%")
        
        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, os.path.join(MODEL_OUTPUT_DIR, 'best_ppo_model.pth'))
            print(f"  -> 保存最佳模型, Reward: {best_reward:.4f}")
    
    # 保存最终模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, os.path.join(MODEL_OUTPUT_DIR, 'ppo_model_final.pth'))
    
    # 保存训练日志
    log_data = {
        'episode_rewards': episode_rewards,
        'episode_profits': episode_profits,
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
            'window_size': WINDOW_SIZE
        }
    }
    
    log_file = os.path.join(TRAINING_LOG_DIR, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n训练完成! 日志保存至: {log_file}")
    
    # 绘制训练曲线
    plot_training_curves(episode_rewards, episode_profits, losses)


def plot_training_curves(rewards, profits, losses):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 奖励曲线
    axes[0, 0].plot(rewards, alpha=0.6)
    axes[0, 0].plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), label='MA-10')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 收益曲线
    axes[0, 1].plot([p*100 for p in profits], alpha=0.6)
    axes[0, 1].plot(np.convolve([p*100 for p in profits], np.ones(10)/10, mode='valid'), label='MA-10')
    axes[0, 1].set_title('Episode Profits (%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Profit (%)')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 损失曲线
    if len(losses) > 0:
        actor_losses = [l[0] for l in losses]
        critic_losses = [l[1] for l in losses]
        axes[1, 0].plot(actor_losses, label='Actor Loss', alpha=0.7)
        axes[1, 0].plot(critic_losses, label='Critic Loss', alpha=0.7)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 累计收益
    cumulative = np.cumsum(profits)
    axes[1, 1].plot(cumulative * 100)
    axes[1, 1].set_title('Cumulative Profit (%)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Profit (%)')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = os.path.join(TRAINING_LOG_DIR, 'training_curves.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {plot_file}")
    
    plt.close()


if __name__ == '__main__':
    train()
