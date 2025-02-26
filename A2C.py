import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.distributions import Categorical
import numpy as np
import random
from torch.nn.functional import mse_loss

# A2C算法注意事项：1.A2C算法的A和C需要两个不同的神经网络（Actor 和 Critic）其参数最好不要共享
# 2.采样时不需要采很长的序列，Agent到达目标就可以结束一个episode
# 3.如果以Agent是否达到目标作为结束条件，容易出现Agent在地图中来回走的情况

class GridWorld:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = set()

        # Generate obstacles (ensure not at start/goal)
        all_cells = [(x, y) for x in range(5) for y in range(5)]
        all_cells.remove(self.start)
        all_cells.remove(self.goal)
        self.obstacles = set(random.sample(all_cells, 6))

    def reset(self):
        self.agent_pos = list(self.start)
        self.done = False
        return self.agent_pos

    def step(self, action):
        if self.done:
            return self.agent_pos, 0, True

        x, y = self.agent_pos
        new_x, new_y = x, y

        # Action mapping: 0=up, 1=down, 2=left, 3=right, 4=stay
        if action == 0:
            new_x = max(x - 1, 0)
        elif action == 1:
            new_x = min(x + 1, 4)
        elif action == 2:
            new_y = max(y - 1, 0)
        elif action == 3:
            new_y = min(y + 1, 4)

        # Check boundary collision
        if (new_x == x and new_y == y) and action != 4:
            reward = -5
        else:
            # Check obstacle collision
            if (new_x, new_y) in self.obstacles:
                reward = -10
            elif (new_x, new_y) == self.goal:
                reward = 10
                self.done = True
            else:
                reward = 0

        self.agent_pos = [new_x, new_y]
        return self.agent_pos.copy(), reward, self.done

    def render(self, episode, actions, states):
        # 可视化环境
        plt.figure(figsize=(5, 5))

        # 绘制目标格
        plt.scatter(self.goal[1], self.goal[0], c='green', s=300, label='Target', marker='s')

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



class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x1 = self.LayerNorm(F.relu(self.fc1(x)))
        x2 = self.LayerNorm(F.relu(self.fc2(x)))
        action_probs = F.softmax(self.actor(x1), dim=-1)
        state_value = self.critic(x2)
        return action_probs, state_value

    #参数初始化
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.constant_(m.bias, 0.1)


# Hyperparameters
INPUT_DIM = 2
HIDDEN_DIM = 32
OUTPUT_DIM = 5
LR = 0.001
GAMMA = 0.95
EPISODES = 2000

# Initialize
env = GridWorld()
model = ActorCritic(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).cuda(0)
model.init_weights()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    # 可视化
    actions = []
    states = []
    t = 0

    while not done and t < 300:
        # 保存状态
        states.append(state)
        # Normalize state
        state_norm = (torch.FloatTensor(state) / 4.0).cuda(0)

        # Get action probabilities and state value
        probs, value = model(state_norm.unsqueeze(0))
        dist = Categorical(probs)
        action = dist.sample()
        actions.append(action.item())

        # Execute action
        next_state, reward, done = env.step(action.item())
        total_reward += reward

        # Calculate TD target
        next_state_norm = (torch.FloatTensor(next_state) / 4.0).cuda(0)
        _, next_value = model(next_state_norm.unsqueeze(0))
        td_target = reward + GAMMA * next_value * (1 - done)
        td_target = td_target.cuda(0)
        td_error = td_target - value
        td_error = td_error.cuda(0)

        # Calculate losses
        actor_loss = -dist.log_prob(action) * td_error.detach().cuda(0)
        critic_loss = mse_loss(value, td_target.detach()).cuda(0)
        loss = actor_loss + critic_loss

        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        t += 1

    if episode % 200 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        env.render(episode, actions, states)

# Test trained model
state = env.reset()
done = False
path = [tuple(state)]
actions = []
states = []

while not done:
    state_norm = (torch.FloatTensor(state) / 4.0).cuda(0)
    with torch.no_grad():
        probs, _ = model(state_norm.unsqueeze(0))
    action = torch.argmax(probs).item()
    next_state, _, done = env.step(action)
    actions.append(action)
    states.append(state)
    path.append(tuple(next_state))
    state = next_state

env.render("Test", actions, states)