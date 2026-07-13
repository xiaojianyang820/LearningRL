import torch
from torch.distributions import Normal
from torch import nn


class TRPOAgent(object):
    """
    TRPOAgent是OnPolicyAgent的具体实现(Trust Region Policy Optimization)。

    策略分布、act()、get_value()全部继承自基类，本类只实现update()：
    自然梯度方向(共轭梯度求解) + KL散度信任域(决定步长) + 回溯线搜索(兜底)。
    策略更新不使用Adam等优化器——步长由信任域约束决定，不是固定学习率；
    只有价值网络用Adam做常规回归。
    """

    def __init__(self, pi_net, v_net, obs_dim, act_dim, device,
                 v_lr=1e-3, gamma=0.99, lam=0.97,
                 delta=0.01, damping_coeff=0.1, cg_iters=10,
                 backtrack_iters=10, backtrack_coeff=0.8, train_v_iters=80):
        """
        初始化TRPOAgent。

        参数:
        - pi_net: 策略网络，输出动作均值
        - v_net: 价值网络
        - obs_dim: 状态维度
        - act_dim: 动作维度
        - device: 计算设备
        - v_lr: 价值网络学习率
        - gamma: 折扣因子
        - lam: GAE-Lambda参数
        - delta: KL散度信任域半径(上限)
        - damping_coeff: Fisher矩阵阻尼系数，防止矩阵病态/奇异
        - cg_iters: 共轭梯度迭代次数
        - backtrack_iters: 回溯线搜索最大尝试次数
        - backtrack_coeff: 线搜索每次的步长衰减系数
        - train_v_iters: 价值网络每个epoch的回归步数
        """
        super().__init__()
        self.device = device
        self.pi_net = pi_net.to(device)
        self.v_net = v_net.to(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam

        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, device=device))

        self.delta = delta
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_v_iters = train_v_iters

        # 策略参数 = pi_net的参数 + 状态无关的log_std(在基类中创建)
        self.pi_params = list(self.pi_net.parameters()) + [self.log_std]

        # TRPO的策略更新不用优化器，只有V网络需要
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=v_lr)

    # ---------------------------------------------------------------- 核心数学工具
    def _distribution(self, obs):
        """
        构造当前策略在给定状态下的动作分布。

        参数:
        - obs: 状态张量

        返回:
        - torch.distributions.Normal 对角高斯分布对象
        """
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    @staticmethod
    def _flat_grad(y, params, retain_graph=False, create_graph=False):
        """
        计算y对params的梯度并拉平成一维向量。

        参数:
        - y: 标量目标
        - params: 参数列表
        - retain_graph: 是否保留计算图(后续还要backward时置True)
        - create_graph: 是否创建高阶导数计算图(算Hessian-vector product时置True)

        返回:
        - 一维梯度向量
        """
        # create_graph=True时必须同时保留原图(PyTorch默认retain_graph跟随create_graph，
        # 这里显式传参会覆盖默认行为，所以要手动做or)，否则第二次backward会报图已释放
        grads = torch.autograd.grad(y, params, retain_graph=retain_graph or create_graph,
                                    create_graph=create_graph)
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _get_flat_params(self):
        """
        取出全部策略参数并拉平成一维向量。

        返回:
        - 一维参数向量
        """
        return torch.cat([p.data.view(-1) for p in self.pi_params])

    def _set_flat_params(self, flat_params):
        """
        把一维参数向量写回各个策略参数(与_get_flat_params互逆)。

        参数:
        - flat_params: 一维参数向量
        """
        idx = 0
        for p in self.pi_params:
            n = p.numel()
            p.data.copy_(flat_params[idx:idx + n].view_as(p))
            idx += n

    def _kl_divergence(self, obs, old_mu, old_std):
        """
        计算 KL(旧策略 || 新策略)，对batch内所有样本取均值。

        参数:
        - obs: 状态batch
        - old_mu: 更新前策略输出的均值(固定不动)
        - old_std: 更新前策略的标准差(固定不动)

        返回:
        - 平均KL散度(标量张量)
        """
        dist = self._distribution(obs)
        old_dist = Normal(old_mu, old_std)
        kl = torch.distributions.kl_divergence(old_dist, dist).sum(axis=-1)
        return kl.mean()

    def _hessian_vector_product(self, obs, old_mu, old_std, v):
        """
        计算Fisher信息矩阵(=KL对参数的Hessian)与向量v的乘积 H@v。

        全程不显式构造H(参数量太大存不下)，用两次反向传播实现：
        第一次对KL求一阶梯度(create_graph=True保留图)，
        第二次对 (grad_kl · v) 再求一次梯度，结果恰好等于 H@v。

        参数:
        - obs: 状态batch
        - old_mu: 旧策略均值
        - old_std: 旧策略标准差
        - v: 被乘的向量

        返回:
        - H@v + damping_coeff*v(阻尼项防止Fisher矩阵退化/奇异)
        """
        kl = self._kl_divergence(obs, old_mu, old_std)
        grad_kl = self._flat_grad(kl, self.pi_params, create_graph=True)
        grad_v = (grad_kl * v).sum()
        hvp = self._flat_grad(grad_v, self.pi_params, retain_graph=True)
        return hvp + self.damping_coeff * v

    def _conjugate_gradient(self, obs, old_mu, old_std, b):
        """
        共轭梯度法近似求解 Hx = b，得到自然梯度方向 x ≈ H^{-1}b。
        全程只需要Hessian-vector product，避免显式存储或求逆H矩阵。

        参数:
        - obs: 状态batch
        - old_mu: 旧策略均值
        - old_std: 旧策略标准差
        - b: 方程右端向量(策略梯度g)

        返回:
        - 近似解x(自然梯度方向)
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(self.cg_iters):
            hvp = self._hessian_vector_product(obs, old_mu, old_std, p)
            alpha = r_dot_old / (torch.dot(p, hvp) + 1e-8)
            x += alpha * p
            r -= alpha * hvp
            r_dot_new = torch.dot(r, r)
            if r_dot_new < 1e-10:
                break
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    # ---------------------------------------------------------------- 更新
    def update(self, data):
        """
        用一个epoch的数据做一次TRPO更新。

        每个epoch只对策略做一次自然梯度更新(不像PPO那样多步)，
        这是TRPO单调改进理论保证成立的前提之一。流程：
        1. 计算代理目标 L=E[ratio*adv] 对策略参数的梯度g
        2. 共轭梯度求自然梯度方向 x ≈ H^{-1}g
        3. 由KL信任域算最大步长 sqrt(2*delta/(x^T H x))
        4. 回溯线搜索找到满足 KL<=delta 且 L确实提升 的步长
        5. V网络独立地做train_v_iters步回归

        参数:
        - data: buffer.get()返回的字典(obs, act, adv, ret, logp)

        返回:
        - 包含surr_obj和v_loss的统计字典
        """
        obs, act, adv = data['obs'], data['act'], data['adv']
        logp_old, ret = data['logp'], data['ret']

        with torch.no_grad():
            old_dist = self._distribution(obs)
            old_mu, old_std = old_dist.mean, old_dist.stddev

        # ---- 1. 代理目标对策略参数的梯度(要最大化L) ----
        dist = self._distribution(obs)
        logp = dist.log_prob(act).sum(axis=-1)
        ratio = torch.exp(logp - logp_old)
        surr_obj = (ratio * adv).mean()
        g = self._flat_grad(surr_obj, self.pi_params)

        # ---- 2. 共轭梯度求自然梯度方向 ----
        x = self._conjugate_gradient(obs, old_mu, old_std, g)

        # ---- 3. KL信任域决定最大步长 ----
        xHx = torch.dot(x, self._hessian_vector_product(obs, old_mu, old_std, x))
        step_size = torch.sqrt(2 * self.delta / (xHx + 1e-8))
        full_step = step_size * x

        # ---- 4. 回溯线搜索 ----
        old_params = self._get_flat_params()
        old_surr_obj = surr_obj.item()
        for i in range(self.backtrack_iters):
            new_params = old_params + full_step * (self.backtrack_coeff ** i)
            self._set_flat_params(new_params)
            with torch.no_grad():
                new_dist = self._distribution(obs)
                new_logp = new_dist.log_prob(act).sum(axis=-1)
                new_ratio = torch.exp(new_logp - logp_old)
                new_surr_obj = (new_ratio * adv).mean().item()
                kl = self._kl_divergence(obs, old_mu, old_std).item()
            if kl <= self.delta and new_surr_obj > old_surr_obj:
                break
            if i == self.backtrack_iters - 1:
                self._set_flat_params(old_params)   # 线搜索全部失败，放弃本次更新

        # ---- 5. V网络回归reward-to-go(与策略更新独立) ----
        v_loss_val = None
        for _ in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            v_loss = ((self.v_net(obs).squeeze(-1) - ret) ** 2).mean()
            v_loss.backward()
            self.v_optimizer.step()
            v_loss_val = v_loss.item()

        return dict(surr_obj=old_surr_obj, v_loss=v_loss_val)

    def act(self, obs, deterministic=False):
        """
        rollout/评估时选择动作，全程不带梯度。

        参数:
        - obs: 当前状态(numpy数组)
        - deterministic: 为True时直接返回分布均值(不采样)，
          对应DDPG评估时noise_scale=0.0的写法

        返回:
        - action: 动作(numpy数组)
        - value: 价值估计 V(s)(标量)
        - logp: 该动作在当前策略下的对数概率(标量)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self._distribution(obs_t)
            action = dist.mean if deterministic else dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
            value = self.v_net(obs_t).squeeze(-1)
        return action.squeeze(0).cpu().numpy(), value.item(), logp.item()

    def get_value(self, obs):
        """
        查询价值网络对某状态的估计，用于轨迹被截断时给buffer.finish_path提供bootstrap值。

        参数:
        - obs: 状态(numpy数组)

        返回:
        - V(s)估计值(标量)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.v_net(obs_t).item()
