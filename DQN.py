import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# 检查可用的GPU数
device_count = torch.cuda.device_count()
print(f"Number of available GPUs: {device_count}")


class GridWorld:
    def __init__(self, size=5, num_obstacles=6):
        self.size = size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.state = (0, 0)  # 起始位置
        self.target = (self.size - 1, self.size - 1)  # 目标位置
        self.obstacles = set()

        while len(self.obstacles) < self.num_obstacles:
            obstacle = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if obstacle != self.target and obstacle != (0, 0):
                self.obstacles.add(obstacle)

        for obs in self.obstacles:
            self.grid[obs] = -10  # 障碍物

        return self.state

    def step(self, action):
        x, y = self.state
        # 定义五个可能的动作
        if action == 0:  # 上
            new_state = (max(x - 1, 0), y)
        elif action == 1:  # 下
            new_state = (min(x + 1, self.size - 1), y)
        elif action == 2:  # 左
            new_state = (x, max(y - 1, 0))
        elif action == 3:  # 右
            new_state = (x, min(y + 1, self.size - 1))
        elif action == 4:  # 不动
            new_state = (x, y)

        reward = self.grid[new_state]
        if new_state == self.target:
            reward = 5
        elif new_state in self.obstacles:
            reward = -10

        self.state = new_state
        done = False  # episode 不主动结束
        return new_state, reward, done

    def render(self, episode, actions, states):
        # 可视化环境
        plt.figure(figsize=(5, 5))

        # 绘制目标格
        plt.scatter(self.target[1], self.target[0], c='green', s=300, label='Target', marker='s')

        # 绘制障碍物
        for obs in self.obstacles:
            plt.scatter(obs[1], obs[0], c='orange', s=300, label='Obstacle', marker='s')

            # 绘制所有状态的动作箭头
        for state, action in zip(states, actions):
            if action == 0:  # 上
                plt.arrow(state[1], state[0], 0, -0.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
            elif action == 1:  # 下
                plt.arrow(state[1], state[0], 0, 0.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
            elif action == 2:  # 左
                plt.arrow(state[1], state[0], -0.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
            elif action == 3:  # 右
                plt.arrow(state[1], state[0], 0.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
            elif action == 4:  # 不动
                plt.scatter(state[1], state[0], c='red', s=100)

        plt.xlim(-0.5, self.size - 0.5)
        plt.ylim(self.size - 0.5, -0.5)
        plt.xticks(np.arange(self.size))
        plt.yticks(np.arange(self.size))
        plt.title(f'Episode {episode}')
        plt.show()


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.8  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.955
        self.model = DQN(state_size, action_size).cuda(0)  # 指定模型使用GPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).cuda()  # 确保状态被转移到GPU
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).cuda()  # 确保状态被转移到GPU
            next_state_tensor = torch.FloatTensor(next_state).cuda()  # 确保下一个状态被转移到GPU

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f.unsqueeze(0).cuda())  # 确保目标在GPU上
            loss.backward()
            self.optimizer.step()

    def reset(self):
        self.memory = deque(maxlen=2000)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


num_episodes = 100
batch_size = 64

env = GridWorld()
state_size = 2  # 状态由x和y坐标组成
action_size = 5  # 动作数量：上、下、左、右、不动
agent = DQNAgent(state_size, action_size)

state = env.reset()
# 训练过程
for e in range(num_episodes):
    state = (0, 0)
    env.state = state
    agent.reset()
    state = np.reshape(state, [1, 2])  # 将状态重塑为适合输入网络的形状

    # 用于存储episode中每个状态的动作
    actions = []
    states = []

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # 存储当前状态和动作
        states.append(state[0])  # 保存当前状态
        actions.append(action)  # 保存当前动作

        next_state = np.reshape(next_state, [1, 2])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        # 在这里不主动结束 episode
        agent.replay(batch_size)

        # 每200个episode更新一次可视化
    if (e + 1) % 20 == 0 or e == 0:
        env.render(e + 1, actions, states)

    print(f"Episode: {e + 1}/{num_episodes}, epsilon: {agent.epsilon:.2}")

print("Training finished.")