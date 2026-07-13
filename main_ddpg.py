import random
import numpy as np
import torch
import gymnasium as gym
from networks import MLPNet
from algos import DDPGAgent
from buffers import UniformOffPolicyBuffer



def main(polyak: float = 0.95, replay_size: int = 10000, total_steps: int = 20000, seed: int = 0,
         env_name: str = "Pendulum-v1", start_steps: int = 1000, action_noise: float = 0.1, epoch_max_len: int = 200,
         update_after: int = 1000, update_every: int = 50, batch_size: int = 100, test_every: int = 200,
         test_episode_num: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建环境
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cn = MLPNet(input_dim=obs_dim + act_dim, hidden_dim=[256, 256], output_dim=1, inner_act_func='relu',
                output_act_func='linear')
    an = MLPNet(input_dim=obs_dim, hidden_dim=[256, 256], output_dim=act_dim, inner_act_func='relu',
                output_act_func='tanh')
    cn_targ = MLPNet(input_dim=obs_dim + act_dim, hidden_dim=[256, 256], output_dim=1, inner_act_func='relu',
                     output_act_func='linear')
    an_targ = MLPNet(input_dim=obs_dim, hidden_dim=[256, 256], output_dim=act_dim, inner_act_func='relu',
                     output_act_func='tanh')
    cn = cn.to(device)
    an = an.to(device)
    cn_targ = cn_targ.to(device)
    an_targ = an_targ.to(device)

    ddpg_agent = DDPGAgent(act_limit=act_limit, critic_net=cn, actor_net=an, critic_net_targ=cn_targ,
                           actor_net_targ=an_targ, polyak=polyak, device=device)
    buffer = UniformOffPolicyBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    test_returns = []  # (step, avg_return)

    obs, _ = env.reset(seed=seed)
    ep_len = 0

    for t in range(total_steps):
        # 前START_STEPS步纯随机探索,之后用策略+高斯噪声
        if t < start_steps:
            act = env.action_space.sample()
        else:
            act = ddpg_agent.act(obs, noise_scale=action_noise)

        next_obs, rew, terminated, truncated, _ = env.step(act)
        ep_len += 1

        done = terminated

        buffer.add(obs, act, rew, next_obs, done)
        obs = next_obs

        if terminated or truncated or ep_len == epoch_max_len:
            obs, _ = env.reset()
            ep_len = 0

        # 按固定频率做UPDATE_EVERY次梯度更新
        if t >= update_after and (t + 1) % update_every == 0:
            for _ in range(update_every):
                batch = buffer.sample(batch_size)
                ddpg_agent.update(batch)

        # 定期评估(不加噪声),观察真实策略水平
        if (t + 1) % test_every == 0:
            avg_ret = evaluate(ddpg_agent, env_name, test_episode_num)
            test_returns.append((t + 1, avg_ret))
            print(f"  step {t + 1:6d} | test avg return: {avg_ret:8.2f}")

    env.close()
    return ddpg_agent, test_returns


# ----------------------------- 评估函数(不加噪声) -----------------------------
def evaluate(agent, env_name, num_episodes, max_ep_len=200):
    env = gym.make(env_name)
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        for _ in range(max_ep_len):
            a = agent.act(obs, noise_scale=0.0)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                break
        returns.append(ep_ret)
    env.close()
    return np.mean(returns)


if __name__ == "__main__":
    main()