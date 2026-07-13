import torch
from torch import nn
from typing import List, Tuple, Dict


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int, inner_act_func: str = 'relu',
                 output_act_func: str = 'linear', bias: bool = True, batch_norm: bool = False):
        """

        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        :param inner_act_func:
        :param output_act_func:
        :param bias:
        :param batch_norm:
        """
        assert inner_act_func in ['relu', 'tanh', 'sigmoid'], "inner_act_func must be one of ['relu', 'tanh', 'sigmoid']"
        assert output_act_func in ['relu', 'tanh', 'sigmoid', 'linear'], "output_act_func must be one of ['relu', 'tanh', 'sigmoid', 'linear']"

        super(MLPNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.inner_act_func = inner_act_func
        self.output_act_func = output_act_func

        self.layers = nn.ModuleList()
        for hd in hidden_dim:
            self.layers.append(nn.Linear(input_dim, hd))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hd))
            if inner_act_func == 'relu':
                self.layers.append(nn.ReLU())
            elif inner_act_func == 'tanh':
                self.layers.append(nn.Tanh())
            elif inner_act_func == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            input_dim = hd

        self.layers.append(nn.Linear(input_dim, output_dim))
        if output_act_func == 'linear':
            pass
        elif output_act_func == 'relu':
            self.layers.append(nn.ReLU())
        elif output_act_func == 'tanh':
            self.layers.append(nn.Tanh())
        elif output_act_func == 'sigmoid':
            self.layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)