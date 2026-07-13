import random
import numpy as np
import torch
import gymnasium as gym
from networks import MLPNet
from algos import TRPOAgent
from buffers import GAEBuffer


def main(steps_per_epoch: int = 4000, epochs: int = 50, seed: int = 0,
         env_name: str = "Pendulum-v1", gamma: float = 0.99, lam: float = 0.97,
         delta: float = 0.01, damping_coeff: float = 0.1, cg_iters: int = 10,
         backtrack_iters: int = 10, backtrack_coeff: float = 0.8, v_lr: float = 1e-3,
         train_v_iters: int = 80, max_ep_len: int = 200,
         test_every_epochs: int = 5, test_episode_num: int = 10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建环境
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # on-policy算法习惯用较小的网络+tanh(spinningup默认(64,64)),
    # 和DDPG的(256,256)+relu是两套不同的经验设置,不是笔误
    pi_net = MLPNet(input_dim=obs_dim, hidden_dim=[64, 64], output_dim=act_dim,
                     inner_act_func='tanh', output_act_func='linear')
    v_net = MLPNet(input_dim=obs_dim, hidden_dim=[64, 64], output_dim=1,
                    inner_act_func='tanh', output_act_func='linear')
    pi_net = pi_net.to(device)
    v_net = v_net.to(device)

    trpo_agent = TRPOAgent(pi_net=pi_net, v_net=v_net, obs_dim=obs_dim, act_dim=act_dim, device=device,
                            v_lr=v_lr, gamma=gamma, lam=lam, delta=delta, damping_coeff=damping_coeff,
                            cg_iters=cg_iters, backtrack_iters=backtrack_iters,
                            backtrack_coeff=backtrack_coeff, train_v_iters=train_v_iters)
    buffer = GAEBuffer(obs_dim=obs_dim, act_dim=act_dim, size=steps_per_epoch, gamma=gamma, lam=lam)

    test_returns = []  # (epoch, avg_return)

    obs, _ = env.reset(seed=seed)
    ep_len = 0

    for epoch in range(epochs):
        # ---- 收集一个epoch的数据(steps_per_epoch步),buffer收满之前不做任何更新 ----
        for t in range(steps_per_epoch):
            act, val, logp = trpo_agent.act(obs)
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)

            next_obs, rew, terminated, truncated, _ = env.step(act_clipped)
            ep_len += 1

            buffer.add(obs, act, rew, val, logp)
            obs = next_obs

            timeout = ep_len == max_ep_len
            epoch_ended = (t + 1) == steps_per_epoch

            if terminated or truncated or timeout or epoch_ended:
                # 只要不是"自然terminated",轨迹就是被截断的,需要V(s)做bootstrap
                if terminated and not (timeout or epoch_ended or truncated):
                    last_val = 0.0
                else:
                    last_val = trpo_agent.get_value(obs)
                buffer.finish_path(last_val)
                obs, _ = env.reset()
                ep_len = 0

        # ---- epoch数据收满,做一次TRPO更新(自然梯度+KL信任域,只更新一次) ----
        data = buffer.get()
        trpo_agent.update(data)

        # 定期评估(用均值动作,不采样噪声),观察真实策略水平
        if (epoch + 1) % test_every_epochs == 0:
            avg_ret = evaluate(trpo_agent, env_name, test_episode_num)
            test_returns.append((epoch + 1, avg_ret))
            print(f"  epoch {epoch + 1:4d} | test avg return: {avg_ret:8.2f}")

    env.close()
    return trpo_agent, test_returns


# ----------------------------- 评估函数(用均值动作,不采样) -----------------------------
def evaluate(agent, env_name, num_episodes, max_ep_len=200):
    env = gym.make(env_name)
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        for _ in range(max_ep_len):
            act, _, _ = agent.act(obs, deterministic=True)
            act_clipped = np.clip(act, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(act_clipped)
            ep_ret += r
            if terminated or truncated:
                break
        returns.append(ep_ret)
    env.close()
    return np.mean(returns)


if __name__ == "__main__":
    main()