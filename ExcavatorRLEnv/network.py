import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class PPO(nn.Module):
    l1_nodes = 64

    def __init__(self, state_dim, action_dim, args):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor_layer = nn.Sequential(
                                    init_(nn.Linear(state_dim, self.l1_nodes)),
                                    nn.Tanh(),
                                    init_(nn.Linear(self.l1_nodes, self.l1_nodes)),
                                    nn.Tanh(),
                                    init_(nn.Linear(self.l1_nodes, action_dim)),
                                    nn.Tanh()).float().to(args.device)

        self.critic_layer = nn.Sequential(
                                    init_(nn.Linear(state_dim, self.l1_nodes)),
                                    nn.Tanh(),
                                    init_(nn.Linear(self.l1_nodes, self.l1_nodes)),
                                    nn.Tanh(),
                                    init_(nn.Linear(self.l1_nodes, 1))).float().to(args.device)

    def forward(self, state):
        return self.actor_layer(state), self.critic_layer(state)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module        