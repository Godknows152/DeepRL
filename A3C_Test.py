import time
import gymnasium as gym
import torch
from torch import nn


# -------------------- Actor网络（仅用于推理） --------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layer(state)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


# -------------------- 可视化最优策略 --------------------
def visualize_policy(model_path, env_name='CartPole-v1'):
    # 加载模型
    model = ActorCritic(state_dim=4, action_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 创建渲染环境
    env = gym.make(env_name, render_mode='human')
    state, info = env.reset()

    total_reward = 0
    while True:
        # 使用确定性策略（选择概率最大的动作）
        state_tensor = torch.FloatTensor(state)
        action_probs, _ = model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.probs.argmax().item()

        # 执行动作并渲染
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

        if terminated or truncated:
            print(f"Episode Reward: {total_reward}")
            total_reward = 0
            state, info = env.reset()

        time.sleep(0.02)  # 控制渲染速度


if __name__ == '__main__':
    visualize_policy('best_model.pth')