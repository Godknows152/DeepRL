import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np


# 定义Actor-Critic网络
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


# 定义A3C Worker线程
class A3CWorker(mp.Process):
    def __init__(self, global_net, optimizer, gamma, n_steps, env_name, worker_id):
        super().__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_steps = n_steps
        self.env = gym.make(env_name)
        self.worker_id = worker_id
        self.local_net = ActorCritic(self.env.observation_space.shape[0],
                                     self.env.action_space.n)

    def run(self):
        state, info = self.env.reset()
        while True:
            states, actions, rewards = [], [], []
            local_net_params = self.global_net.state_dict()
            self.local_net.load_state_dict(local_net_params)

            # 收集n步经验
            for _ in range(self.n_steps):
                state_tensor = torch.FloatTensor(state)
                action_probs, _ = self.local_net(state_tensor)

                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()  # 采样动作

                next_state, reward, done, _, info = self.env.step(action.item())

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)

                state = next_state
                if done:
                    state, info = self.env.reset()
                    break

            # 计算n步回报和优势
            R = 0 if done else self.local_net(torch.FloatTensor(state))[1].item()
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

            # 价值损失
            value_loss = 0.5 * (state_values.squeeze() - returns_tensor).pow(2).mean()

            # 总损失
            total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            # 异步更新全局网络
            self.optimizer.zero_grad()
            total_loss.backward()
            for local_param, global_param in zip(self.local_net.parameters(),
                                                 self.global_net.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad
            self.optimizer.step()


# 主函数
if __name__ == '__main__':
    mp.set_start_method('spawn')
    env_name = 'CartPole-v1'
    state_dim = 4
    action_dim = 2
    gamma = 0.99
    n_steps = 5
    num_workers = 4

    global_net = ActorCritic(state_dim, action_dim)
    global_net.share_memory()
    optimizer = optim.Adam(global_net.parameters(), lr=1e-3)

    workers = [A3CWorker(global_net, optimizer, gamma, n_steps, env_name, i)
               for i in range(num_workers)]

    for w in workers:
        w.start()
    for w in workers:
        w.join()