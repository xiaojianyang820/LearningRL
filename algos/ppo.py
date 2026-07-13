from .trpo import TRPOAgent
import torch
from torch import nn
import numpy as np
#test

class PPOAgent(TRPOAgent):
    """
    PPOAgent是OnPolicyAgent的具体实现(PPO-Clip版本)。

    策略分布、act()、get_value()全部继承自基类，本类只实现update()：
    用裁剪代理目标(clipped surrogate objective)代替TRPO的
    KL信任域+共轭梯度+线搜索，同一批数据可以做多步梯度更新，
    近似KL超标时提前停止。实现简单得多，实践中稳定性与TRPO相当。
    """

    def __init__(self, pi_net, v_net, obs_dim, act_dim, device,
                 pi_lr=3e-4, v_lr=1e-3, gamma=0.99, lam=0.97,
                 clip_ratio=0.2, train_pi_iters=80, train_v_iters=80,
                 target_kl=0.01):
        """
        初始化PPOAgent。

        参数:
        - pi_net: 策略网络，输出动作均值
        - v_net: 价值网络
        - obs_dim: 状态维度
        - act_dim: 动作维度
        - device: 计算设备
        - pi_lr: 策略学习率(PPO与TRPO不同，策略用Adam做常规梯度更新)
        - v_lr: 价值网络学习率
        - gamma: 折扣因子
        - lam: GAE-Lambda参数
        - clip_ratio: 概率比裁剪范围，ratio被限制在[1-clip_ratio, 1+clip_ratio]
        - train_pi_iters: 每个epoch策略最多做几步梯度更新
        - train_v_iters: 每个epoch价值网络的回归步数
        - target_kl: 近似KL超过1.5倍该值时提前停止策略更新
        """
        super(PPOAgent, self).__init__(pi_net, v_net, obs_dim, act_dim, device, gamma=gamma, lam=lam)
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        # PPO用Adam更新策略：pi_net的参数 + 状态无关的log_std(在基类中创建)
        self.pi_optimizer = torch.optim.Adam(list(self.pi_net.parameters()) + [self.log_std], lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=v_lr)

    # ---------------------------------------------------------------- 更新
    def update(self, data):
        """
        用一个epoch的数据做PPO-Clip更新。

        与TRPO最大的区别：同一批数据反复用来做多步(最多train_pi_iters步)梯度更新，
        靠clip(而非硬性KL约束)防止单步更新幅度过大；一旦近似KL超过
        1.5*target_kl就提前停止。这是PPO"实践上足够稳"但没有TRPO那种
        理论单调改进保证的地方。

        参数:
        - data: buffer.get()返回的字典(obs, act, adv, ret, logp)

        返回:
        - 包含pi_loss、v_loss和stop_iter(实际执行的策略更新步数)的统计字典
        """
        obs, act, adv = data['obs'], data['act'], data['adv']
        logp_old, ret = data['logp'], data['ret']

        # ---- Policy: 多步clip更新 ----
        pi_loss_val, stop_iter = None, self.train_pi_iters
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            dist = self._distribution(obs)
            logp = dist.log_prob(act).sum(axis=-1)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            pi_loss = -torch.min(ratio * adv, clip_adv).mean()

            # 近似KL(不是真KL散度，是便宜的替代估计量，用来判断是否走远了)
            approx_kl = (logp_old - logp).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                stop_iter = i
                break

            pi_loss.backward()
            self.pi_optimizer.step()
            pi_loss_val = pi_loss.item()

        # ---- Value: 多步回归，与policy更新相互独立 ----
        v_loss_val = None
        for _ in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            v_loss = ((self.v_net(obs).squeeze(-1) - ret) ** 2).mean()
            v_loss.backward()
            self.v_optimizer.step()
            v_loss_val = v_loss.item()

        return dict(pi_loss=pi_loss_val, v_loss=v_loss_val, stop_iter=stop_iter)

