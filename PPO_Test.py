import gymnasium as gym
import torch
from torch import nn


# 加载训练好的模型
class Actor(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


def test_model(model_path):
    # 初始化环境和模型
    env = gym.make('CartPole-v1', render_mode='human')
    model = Actor()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    state, info = env.reset()
    episode_reward = 0

    while True:
        # 选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()  # 选择确定性动作
        # 执行动作
        next_state, reward, done, truncated, info= env.step(action)
        episode_reward += reward
        state = next_state

        if done or truncated:
            print(f"Episode :完成一轮! 累计奖励: {episode_reward}")
            episode_reward = 0
            state, info = env.reset()



if __name__ == "__main__":
    # 使用训练好的最佳模型
    test_model("PPO_Mode/best_model_CartPole-v1.pth")