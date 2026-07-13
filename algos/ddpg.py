import numpy as np
from typing import List, Tuple, Dict, Callable
from torch import nn
import torch
from torch.nn import functional as F

class DDPGAgent(object):
    def __init__(self, act_limit: float, critic_net: nn.Module, actor_net: nn.Module, critic_net_targ: nn.Module,
                 actor_net_targ: nn.Module,
                 polyak: float = 0.95, gamma:float = 0.99, actor_lr: float = 1e-3, critic_lr: float = 1e-3,
                 device: str = 'cpu'):
        self.act_limit = act_limit
        self.polyak = polyak
        self.device = device
        self.critic_net = critic_net
        self.actor_net = actor_net
        self.critic_net_targ = critic_net_targ
        self.actor_net_targ = actor_net_targ
        self.gamma = gamma

        self.critic = self._generate_critic(critic_net)
        self.actor = self._generate_actor(actor_net)
        self.critic_targ = self._generate_critic(critic_net_targ)
        self.actor_targ = self._generate_actor(actor_net_targ)

        # 同步主网络和目标网络的模型参数
        self._copy_main_params_to_targ_params()
        # 关闭目标网络模型参数的梯度计算功能
        for p in critic_net_targ.parameters():
            p.requires_grad = False
        for p in actor_net_targ.parameters():
            p.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)

    def act(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            a = self.actor(obs_tensor).cpu().numpy()
        a += noise_scale * self.act_limit * np.random.randn(a.shape[-1])
        a = np.array(np.clip(a, -self.act_limit, self.act_limit))
        return a

    def update(self, batch: Dict[str, np.ndarray]) -> Tuple[float, float]:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["obs2"], batch["done"]

        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        act = torch.as_tensor(act, device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
        done = torch.as_tensor(done, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            next_a = self.actor_targ(next_obs)
            next_q = self.critic_targ(torch.cat([next_obs, next_a], dim=1))
            target_q = rew + self.gamma * next_q * (1 - done)

        current_q = self.critic(torch.cat([obs, act], dim=1))
        loss_critic = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # 关闭掉Critic网络的参数梯度计算
        for p in self.critic_net.parameters():
            p.requires_grad = False

        loss_actor = -self.critic(torch.cat([obs, self.actor(obs)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        for p in self.critic_net.parameters():
            p.requires_grad = True

        # ---- Target网络软更新(Polyak averaging) ----
        with torch.no_grad():
            for p, p_targ in zip(self.critic_net.parameters(), self.critic_net_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.actor_net.parameters(), self.actor_net_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return loss_critic.item(), loss_actor.item()


    @staticmethod
    def _generate_critic(critic_net: nn.Module) -> Callable:
        return lambda x : critic_net(x).squeeze(-1)

    def _generate_actor(self, actor_net: nn.Module) -> Callable:
        return lambda x: actor_net(x) * self.act_limit

    def _copy_main_params_to_targ_params(self):
        self.critic_net_targ.load_state_dict(self.critic_net.state_dict())
        self.actor_net_targ.load_state_dict(self.actor_net.state_dict())




if __name__ == '__main__':
    from networks import MLPNet
    obs_dim = 4
    act_dim = 1
    device = 'cpu'
    cn = MLPNet(input_dim=obs_dim + act_dim, hidden_dim=[64, 128, 64], output_dim=1, inner_act_func='relu',
                        output_act_func='linear')
    an = MLPNet(input_dim=obs_dim, hidden_dim=[64, 128, 64], output_dim=act_dim, inner_act_func='relu',
                       output_act_func='tanh')
    cn_targ = MLPNet(input_dim=obs_dim + act_dim, hidden_dim=[64, 128, 64], output_dim=1, inner_act_func='relu',
                        output_act_func='linear')
    an_targ = MLPNet(input_dim=obs_dim, hidden_dim=[64, 128, 64], output_dim=act_dim, inner_act_func='relu',
                       output_act_func='tanh')
    cn = cn.to(device)
    an = an.to(device)
    cn_targ = cn_targ.to(device)
    an_targ = an_targ.to(device)

    ddpg_agent = DDPGAgent(act_limit=5, critic_net=cn, actor_net=an, critic_net_targ=cn_targ, actor_net_targ=an_targ)
    obs = torch.Tensor(np.random.uniform(-1, 1, (4, obs_dim)))
    act = torch.Tensor(np.random.uniform(-1, 1, (4, act_dim)))
    print(ddpg_agent.critic(torch.concatenate([obs, act], dim=1)))
    print(ddpg_agent.actor(obs))

    cn_param = cn.state_dict()['layers.0.weight'][0]
    cn_targ_param = cn_targ.state_dict()['layers.0.weight'][0]
    print(cn_param, cn_targ_param)



