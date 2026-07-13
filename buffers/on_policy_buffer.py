"""
On-policy 经验缓冲区 (GAEBuffer)
供 TRPO / PPO 等 on-policy 算法共用，与 off_policy_buffer.py 里的
UniformOffPolicyBuffer(供DDPG用)是并列关系，风格保持一致。

核心区别：
- off-policy buffer 是环形队列，可以反复随机采样(sample)旧数据
- on-policy buffer 是"一次性"的，每个epoch收满就必须get()取走用掉，
  然后清零重新收集，因为TRPO/PPO的更新公式要求数据来自当前策略
"""
from abc import ABC, abstractmethod
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discount_cumsum(x, discount):
    """
    从后往前算折扣累计和，不依赖scipy。
    输入 [x0, x1, x2]，输出 [x0+d*x1+d^2*x2, x1+d*x2, x2]
    用来同时计算 reward-to-go 和 GAE-Lambda 优势
    """
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(x))):
        running = x[i] + discount * running
        out[i] = running
    return out


class OnPolicyBuffer(ABC):
    @abstractmethod
    def add(self, state, action, reward, value, logp):
        pass

    @abstractmethod
    def finish_path(self, last_val):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def size(self):
        pass


class GAEBuffer(OnPolicyBuffer):
    """
    存储一个epoch内所有轨迹的 (s, a, r, v, logp)。
    轨迹结束时调用finish_path()，用GAE-Lambda把整条轨迹的优势算出来；
    整个epoch收满后调用get()一次性取出全部数据(并做优势归一化)。
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)  # reward-to-go，训练V网络的回归目标
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def add(self, obs, act, rew, val, logp):
        """每和环境交互一步调用一次，存一条transition"""
        assert self.ptr < self.max_size, "buffer已满，应该在epoch结束时调用get()清空"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        """
        一条轨迹结束时调用(自然terminated，或被truncated/epoch截断)。
        last_val: 轨迹是被截断的(没有真正到达终止状态)时，传入V(s_last)做bootstrap；
                  自然终止(terminated=True)时传0，因为终止状态之后没有未来奖励了。
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)，再做折扣累加
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # reward-to-go，用来回归V网络（不是优势，是真实的折扣回报估计）
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        epoch结束时调用，取出整个buffer的数据并重置指针。
        优势做归一化(均值0标准差1)——这是on-policy方法里公认重要的稳定训练的技巧。
        """
        assert self.ptr == self.max_size, "get()前buffer必须被填满"
        self.ptr, self.path_start_idx = 0, 0

        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        adv_normalized = (self.adv_buf - adv_mean) / (adv_std + 1e-8)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=adv_normalized, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for k, v in data.items()}

    def size(self):
        return self.ptr