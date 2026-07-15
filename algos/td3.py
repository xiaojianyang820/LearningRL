"""
TD3Agent (Twin Delayed DDPG)
在DDPG基础上加三个技巧(对应 spinningup TD3文档 Trick One/Two/Three):

1. Clipped Double-Q Learning: 学两个critic(q1,q2)，算target时取两者较小值，
   抑制DDPG里常见的Q值高估问题(Q-function overestimation)。
2. Delayed Policy Updates: policy网络和target网络的更新频率比critic慢，
   每policy_delay次critic更新才做一次policy+target更新(论文推荐2)。
3. Target Policy Smoothing: 算target时给target action加裁剪过的高斯噪声，
   让Q函数在相近动作上更平滑，防止策略钻Q函数误差的空子。

接口风格对齐DDPGAgent：网络在main脚本里用networks.MLPNet搭好，从外部传入，
唯一区别是critic变成两份(critic_net1/2 + 对应的target)。
"""

import numpy as np
import torch
import torch.nn.functional as F


class TD3Agent:
    def __init__(self, act_limit, act_dim, actor_net, actor_net_targ,
                 critic_net1, critic_net2, critic_net_targ1, critic_net_targ2,
                 device, polyak=0.995, gamma=0.99, pi_lr=1e-3, q_lr=1e-3,
                 target_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.device = device
        self.polyak = polyak
        self.gamma = gamma
        self.target_noise = target_noise   # target policy smoothing噪声的标准差 sigma
        self.noise_clip = noise_clip       # 该噪声被裁剪到 [-noise_clip, noise_clip]
        self.policy_delay = policy_delay   # 每几次critic更新才做一次policy+target更新

        self.actor_net = actor_net.to(device)
        self.actor_net_targ = actor_net_targ.to(device)
        self.critic_net1 = critic_net1.to(device)
        self.critic_net2 = critic_net2.to(device)
        self.critic_net_targ1 = critic_net_targ1.to(device)
        self.critic_net_targ2 = critic_net_targ2.to(device)

        # target网络初始化为和main网络一样的参数(即便main脚本已经复制过,这里再做一次也无害)
        self.actor_net_targ.load_state_dict(self.actor_net.state_dict())
        self.critic_net_targ1.load_state_dict(self.critic_net1.state_dict())
        self.critic_net_targ2.load_state_dict(self.critic_net2.state_dict())

        # target网络不需要梯度
        for p in self.actor_net_targ.parameters():
            p.requires_grad = False
        for p in self.critic_net_targ1.parameters():
            p.requires_grad = False
        for p in self.critic_net_targ2.parameters():
            p.requires_grad = False

        self.pi_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=pi_lr)
        # 两个critic合并成一个optimizer,critic_loss=mse(q1)+mse(q2)一次backward同时更新两个网络
        self.q_optimizer = torch.optim.Adam(
            list(self.critic_net1.parameters()) + list(self.critic_net2.parameters()), lr=q_lr)

        self._update_counter = 0   # 记录update()被调用了多少次,用来判断该不该做延迟更新

    def act(self, obs, noise_scale=0.0):
        """
        与DDPG完全一样:确定性策略输出动作 + 训练时加均值0的高斯探索噪声。
        noise_scale=0.0时就是纯确定性策略,用于评估。
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor_net(obs_t).cpu().numpy()
        a = a + noise_scale * self.act_limit * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def update(self, batch):
        """
        batch来自buffer.sample(batch_size)，字段与DDPG完全一样:obs, act, rew, obs2, done。
        每次调用都会更新两个critic；只有第policy_delay的倍数次调用才会更新actor和target网络。
        """
        obs, act, rew = batch['obs'], batch['act'], batch['rew']
        obs2, done = batch['obs2'], batch['done']

        # ---- 1. Target policy smoothing + clipped double-Q: 算target ----
        with torch.no_grad():
            eps = torch.randn_like(act) * self.target_noise
            eps = torch.clamp(eps, -self.noise_clip, self.noise_clip)
            act2 = self.actor_net_targ(obs2) + eps
            act2 = torch.clamp(act2, -self.act_limit, self.act_limit)   # 裁剪到合法动作范围

            q1_targ = self.critic_net_targ1(torch.cat([obs2, act2], dim=-1)).squeeze(-1)
            q2_targ = self.critic_net_targ2(torch.cat([obs2, act2], dim=-1)).squeeze(-1)
            q_targ = torch.min(q1_targ, q2_targ)          # 取较小值,抑制Q值高估
            backup = rew + self.gamma * (1 - done) * q_targ

        # ---- 2. 更新两个critic(每次update()都做,不受policy_delay影响) ----
        q1 = self.critic_net1(torch.cat([obs, act], dim=-1)).squeeze(-1)
        q2 = self.critic_net2(torch.cat([obs, act], dim=-1)).squeeze(-1)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        self._update_counter += 1
        actor_loss_val = None

        # ---- 3. 延迟的policy更新 + target软更新 ----
        if self._update_counter % self.policy_delay == 0:
            # 策略只用q1求梯度(spinningup就是只用q1,不是取q1/q2的min或平均)，
            # 更新policy时critic1的参数要临时冻结,只让梯度流向actor
            for p in self.critic_net1.parameters():
                p.requires_grad = False
            actor_loss = -self.critic_net1(
                torch.cat([obs, self.actor_net(obs)], dim=-1)).squeeze(-1).mean()
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()
            for p in self.critic_net1.parameters():
                p.requires_grad = True
            actor_loss_val = actor_loss.item()

            # target网络软更新(Polyak averaging)：actor + 两个critic都要更新
            with torch.no_grad():
                for p, p_targ in zip(self.actor_net.parameters(), self.actor_net_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
                for p, p_targ in zip(self.critic_net1.parameters(), self.critic_net_targ1.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
                for p, p_targ in zip(self.critic_net2.parameters(), self.critic_net_targ2.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(critic_loss=critic_loss.item(), actor_loss=actor_loss_val)