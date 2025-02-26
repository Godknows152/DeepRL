import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np



# -------------------- Actor-Critic 网络 --------------------
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


# -------------------- A3C Worker 进程 --------------------
class A3CWorker(mp.Process):
    def __init__(self, global_net, optimizer, gamma, n_steps, env_name, worker_id, reward_queue):
        super().__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.alpha = 0.01
        self.n_steps = n_steps
        self.env = gym.make(env_name)
        self.worker_id = worker_id
        self.local_net = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n)
        self.reward_queue = reward_queue  # 新增：奖励队列用于保存模型

    def run(self):
        episode_reward = 0
        state, info = self.env.reset()
        while True:
            states, actions, rewards = [], [], []
            self.local_net.load_state_dict(self.global_net.state_dict())

            for _ in range(self.n_steps):
                state_tensor = torch.FloatTensor(state)
                action_probs, _ = self.local_net(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

                if terminated or truncated:
                    # 发送当前episode的总奖励到主进程
                    self.reward_queue.put(episode_reward)
                    episode_reward = 0
                    state, info = self.env.reset()
                    break

            R = 0 if terminated else self.local_net(torch.FloatTensor(state))[1].item()
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(actions)
            returns_tensor = torch.FloatTensor(returns)

            # 计算损失
            action_probs, state_values = self.local_net(states_tensor)
            advantages = returns_tensor - state_values.squeeze().detach()

            # 策略损失（含熵正则化）
            dist = torch.distributions.Categorical(action_probs)
            policy_loss = - (dist.log_prob(actions_tensor) * advantages).mean()
            entropy_loss = - dist.entropy().mean()
            action_loss = policy_loss + self.alpha * entropy_loss

            # 价值损失
            value_loss = 0.5 * (state_values.squeeze() - returns_tensor).pow(2).mean()

            # 总损失
            total_loss = (action_loss + 0.5 * value_loss)

            # 异步更新全局网络
            self.optimizer.zero_grad()
            total_loss.backward()
            for local_param, global_param in zip(self.local_net.parameters(),
                                                 self.global_net.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
            self.optimizer.step()


# -------------------- 主函数（新增模型保存逻辑） --------------------
if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_name = 'CartPole-v1'
    state_dim = 4
    action_dim = 2
    gamma = 0.99
    n_steps = 5
    num_workers = 4

    # 初始化全局网络和优化器
    global_net = ActorCritic(state_dim, action_dim)
    global_net.share_memory()
    optimizer = optim.Adam(global_net.parameters(), lr=1e-3)

    # 新增：创建奖励队列和最佳模型跟踪
    reward_queue = mp.Queue()
    best_reward = -float('inf')

    # 创建并启动Worker进程
    workers = [A3CWorker(global_net, optimizer, gamma, n_steps, env_name, i, reward_queue)
               for i in range(num_workers)]
    for w in workers:
        w.start()

    # 新增：主进程监控奖励并保存最佳模型
    try:
        while True:
            if not reward_queue.empty():
                current_reward = reward_queue.get()
                if current_reward > best_reward:
                    best_reward = current_reward
                    torch.save(global_net.state_dict(), 'best_model.pth')
                    print(f"Saved best model with reward: {best_reward}")
    except KeyboardInterrupt:
        # 训练完成后按Ctrl+C退出
        pass

    # 等待所有Worker结束
    for w in workers:
        w.join()