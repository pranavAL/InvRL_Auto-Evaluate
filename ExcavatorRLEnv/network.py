import torch.nn as nn

class PPO(nn.Module):
    l1_nodes = 128

    def __init__(self, state_dim, action_dim, args):
        super().__init__()

        self.actor_layer = nn.Sequential(
                                    nn.Linear(state_dim, self.l1_nodes),
                                    nn.ELU(),
                                    nn.Linear(self.l1_nodes, action_dim),
                                    nn.Tanh()).float().to(args.device)

        self.critic_layer = nn.Sequential(
                                    nn.Linear(state_dim, self.l1_nodes),
                                    nn.ELU(),
                                    nn.Linear(self.l1_nodes, 1)).float().to(args.device)

    def forward(self, state):
        return self.actor_layer(state), self.critic_layer(state)
