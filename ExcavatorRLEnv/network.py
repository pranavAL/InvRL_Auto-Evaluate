import torch.nn as nn

class PPO(nn.Module):
    l1_nodes = 64

    def __init__(self, state_dim, action_dim, args):
        super().__init__()

        self.feature_layer = nn.Sequential(
                                    nn.Linear(state_dim, self.l1_nodes),
                                    nn.Tanh(),
                                    nn.Linear(self.l1_nodes, self.l1_nodes),
                                    nn.Tanh())
                
        self.actor_layer = nn.Sequential(nn.Linear(self.l1_nodes, action_dim),
                                        nn.Tanh())

        self.critic_layer = nn.Sequential(nn.Linear(self.l1_nodes, 1))                                        

    def forward(self, state):
        features = self.feature_layer(state)
        return self.actor_layer(features), self.critic_layer(features)
