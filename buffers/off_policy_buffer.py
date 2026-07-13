# 导入抽象类
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
import numpy as np

class OffPolicyBuffer(ABC):
    """
    OffPolicyBuffer是一个抽象类，用于定义离策略强化学习中经验回放缓冲区的接口。
    该类提供了存储和采样经验的基本方法，具体实现需要在子类中完成。
    """

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """
        将一个经验元组添加到缓冲区中。

        参数:
        - state: 当前状态
        - action: 当前动作
        - reward: 当前奖励
        - next_state: 下一状态
        - done: 是否结束标志
        """

    @abstractmethod
    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验。

        参数:
        - batch_size: 采样的批量大小

        返回:
        - 一个包含状态、动作、奖励、下一状态和结束标志的批量数据
        """

    @abstractmethod
    def size(self):
        """
        返回缓冲区中当前存储的经验数量。

        返回:
        - 缓冲区中经验的数量
        """


class UniformOffPolicyBuffer(OffPolicyBuffer):
    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        """
        初始化一个offpolicybuffer

        :param obs_dim: int,
            状态的维度
        :param act_dim: int,
            动作的维度
        :param size: int,
            缓冲区的大小
        """
        # s的存储列表
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: float) -> None:
        """
        将一个经验元组添加到缓冲区中。

        :param obs: np.ndarray,
            当前状态
        :param act:
        :param rew:
        :param next_obs:
        :param done:
        :return:
        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idx],
            obs2=self.next_obs_buf[idx],
            act=self.act_buf[idx],
            rew=self.rew_buf[idx],
            done=self.done_buf[idx],
        )
        return batch

    def size(self) -> int:
        return self.size


if __name__ == '__main__':
    # 如果没有把全部抽象方法实现，无法构造它的实例。
    buffer = UniformOffPolicyBuffer(obs_dim=4, act_dim=3, size=10)
