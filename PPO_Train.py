import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


# 超参数配置
class Config:
    env_name = "CartPole-v1"
    lr = 3e-4
    gamma = 0.99  # 折扣因子
    eps_clip = 0.2  # PPO clip参数
    K_epochs = 4  # 每批数据训练的epoch数
    batch_size = 64  # 小批量大小
    hidden_dim = 128  # 网络隐藏层维度
    update_interval = 2000  # 每收集多少步更新一次
    max_episodes = 600  # 最大训练episode数
    max_steps = 500  # 每个episode最大步数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 策略网络（Actor）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# 价值网络（Critic）
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


# PPO智能体
class PPOAgent:
    def __init__(self, config):
        self.cfg = config
        self.env = gym.make(config.env_name)

        # 初始化网络
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.actor = Actor(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.critic = Critic(state_dim, config.hidden_dim).to(config.device)
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=config.lr)

        # 经验缓存
        self.memory = deque()
        self.total_steps = 0
        self.best_reward = 0

    # 选择动作（带概率）
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.cfg.device)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    # 计算折扣回报
    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.cfg.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    # PPO更新
    def update(self):
        # 将经验转换为张量
        states = torch.FloatTensor(np.array([t[0] for t in self.memory])).to(self.cfg.device)
        actions = torch.LongTensor(np.array([t[1] for t in self.memory])).to(self.cfg.device)
        old_log_probs = torch.FloatTensor(np.array([t[2] for t in self.memory])).to(self.cfg.device)
        rewards = np.array([t[3] for t in self.memory])
        next_states = torch.FloatTensor(np.array([t[4] for t in self.memory])).to(self.cfg.device)
        dones = np.array([t[5] for t in self.memory])

        # 计算折扣回报和优势
        returns = self.compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns).to(self.cfg.device)

        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # 优化K个epoch
        for _ in range(self.cfg.K_epochs):
            for idx in range(0, len(states), self.cfg.batch_size):
                # 小批量采样
                batch_idx = slice(idx, idx + self.cfg.batch_size)
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]

                # 计算当前策略的概率
                action_probs = self.actor(b_states)
                dist = Categorical(action_probs)
                b_new_log_probs = dist.log_prob(b_actions)

                # 重要性采样比率
                ratios = torch.exp(b_new_log_probs - b_old_log_probs)

                # 计算价值估计
                V = self.critic(b_states).squeeze()
                advantages = b_returns - V.detach()

                # 计算损失函数
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(V, b_returns)

                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss

                # 梯度下降
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # 清空经验缓存
        self.memory.clear()

    # 训练函数
    def train(self):
        print("开始训练...")
        for ep in range(self.cfg.max_episodes):
            state, info = self.env.reset()
            ep_reward = 0
            for _ in range(self.cfg.max_steps):
                # 选择动作
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, info = self.env.step(action)

                # 存储经验
                self.memory.append((state, action, log_prob, reward, next_state, done))
                self.total_steps += 1

                state = next_state
                ep_reward += reward

                # 定期更新网络
                if self.total_steps % self.cfg.update_interval == 0 and len(self.memory) > 0:
                    self.update()

                if done:
                    break

            # 记录最佳模型
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                torch.save(self.actor.state_dict(), f"PPO_Mode/best_model_{self.cfg.env_name}.pth")

            # 打印训练进度
            if ep % 20 == 0:
                print(f"Episode {ep}, Reward: {ep_reward}, Best Reward: {self.best_reward}")

            # 提前停止条件
            if self.best_reward >= 500:
                print("已达成完美表现！")
                break

        self.env.close()


if __name__ == "__main__":
    config = Config()
    agent = PPOAgent(config)
    agent.train()