import os
import time
import wandb
import torch
import numpy as np
import torch.nn as nn
from network import PPO
from arguments import get_args
from replay_buffer import Memory
from torch.distributions import MultivariateNormal

class Agent:
    def __init__(self, args):
        self.policy_clip = args.policy_clip
        self.value_clip = args.value_clip
        self.entropy_coef = args.weight_entropy
        self.vf_loss_coef = args.weight_for_value
        self.PPO_epochs = args.ppo_epochs
        self.lr_actor = args.lr_act
        self.lr_critic = args.lr_crit
        self.is_training_mode = True
        self.state_dim = 10
        self.action_dim = 4
        self.action_std = 0.6
        self.gamma = args.gamma
        self.lam = args.lam
        self.args = args
        self.loss_epoch = 0

        self.policy = PPO(self.state_dim, self.action_dim, self.args).to(self.args.device)
        self.policy_old = PPO(self.state_dim, self.action_dim, self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.betas = (0.9, 0.999)

        self.policy_optimizer = torch.optim.Adam([
                                    {'params': self.policy.actor_layer.parameters(), 'lr':self.lr_actor, 'betas':self.betas},
                                    {'params': self.policy.critic_layer.parameters(), 'lr':self.lr_critic, 'betas':self.betas}
                                    ])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=5000, gamma=0.96)

        self.memory = Memory()
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(self.args.device)
        self.cov_mat = torch.diag_embed(self.action_var).to(self.args.device).detach()
        self.mseloss = nn.MSELoss()

    def save_eps(self, state, reward, action, done, next_state):
        self.memory.save_eps(state, reward, action, done, next_state)

    def evaluate_loss(self, states, actions, rewards, next_states, dones):
        self.loss_epoch += 1
        action_mean, values  = self.policy(states)
        old_action_mean, old_values = self.policy_old(states)
        _, next_values  = self.policy(next_states)

        old_values = old_values.detach()

        distribution = MultivariateNormal(action_mean, self.cov_mat)
        dist_entropy = distribution.entropy().to(self.args.device)

        advantages = self.generalized_advantage_estimation(values, rewards, next_values, dones).detach()
        returns = self.temporal_difference(rewards, next_values, dones).detach()

        critic_loss = self.mseloss(returns, values) * 0.5

        logprobs = distribution.log_prob(actions).float().to(self.args.device)
        old_distribution = MultivariateNormal(old_action_mean, self.cov_mat)
        old_logprobs = old_distribution.log_prob(actions).float().to(self.args.device).detach()

        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        pg_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy

        loss = pg_loss + critic_loss

        self.meta_data['Advantage'].append(advantages.mean().item())
        self.meta_data['Entropy'].append(dist_entropy.mean().item())
        self.meta_data['TD'].append(returns.mean().item())
        self.meta_data['Critic_Loss'].append(critic_loss.mean().item())
        self.meta_data['KL'].append(ratios.mean().item())
        self.meta_data['Policy_Loss'].append(pg_loss.mean().item())

        return loss

    def act(self, state, is_training=True):

        self.is_training_mode = True
        state = torch.FloatTensor(state).to(self.args.device)
        action_mean, _ = self.policy_old(state)

        if self.is_training_mode:
            distribution = MultivariateNormal(action_mean, self.cov_mat)
            action = distribution.sample().float().to(self.args.device)
            return np.clip(action.cpu().numpy(), -1, 1)
        else:
            action = action_mean.cpu().detach().numpy()
            return np.clip(action,-1, 1)

    def update_ppo(self):
        length = len(self.memory.buffer["states"])

        states = torch.FloatTensor(self.memory.buffer["states"]).to(self.args.device).detach()
        actions = torch.FloatTensor(self.memory.buffer["actions"]).to(self.args.device).detach()
        rewards = torch.FloatTensor(self.memory.buffer["rewards"]).view(length, 1).to(self.args.device).detach()
        dones = torch.FloatTensor(self.memory.buffer["dones"]).view(length, 1).to(self.args.device).detach()
        next_states = torch.FloatTensor(self.memory.buffer["next_states"]).to(self.args.device).detach()

        self.meta_data = {'Total_Loss':[],'Value':[],'Advantage':[],'Entropy':[],'TD':[],'Critic_Loss':[],'KL':[],'Policy_Loss':[], 'Learning_Rate':[]}

        for epoch in range(self.PPO_epochs):

            loss = self.evaluate_loss(states, actions, rewards, next_states, dones)
            self.policy_optimizer.zero_grad()

            loss.mean().backward()
            self.policy_optimizer.step()
            self.scheduler.step()

        self.memory.deleteBuffer()
        self.policy_old.load_state_dict(self.policy.state_dict())

        wandb.log({'Advantage' : np.mean(self.meta_data['Advantage'])})
        wandb.log({'Entropy' :np.mean(self.meta_data['Entropy'])})
        wandb.log({'TD': np.mean(self.meta_data['TD'])})
        wandb.log({'Critic_Loss': np.mean(self.meta_data['Critic_Loss'])})
        wandb.log({'KL': np.mean(self.meta_data['KL'])})
        wandb.log({'Policy_Loss': np.mean(self.meta_data['Policy_Loss'])})
        wandb.log({'Actor Learning Rate': self.policy_optimizer.param_groups[0]['lr']})
        wandb.log({'Critic Learning Rate': self.policy_optimizer.param_groups[1]['lr']})

    def generalized_advantage_estimation(self, values, rewards, next_value, done):
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + self.gamma * self.lam * gae
            returns.insert(0, gae)

        return torch.stack(returns)

    def temporal_difference(self, rewards, next_values, dones):
        TD = rewards + self.gamma * next_values * (1 - dones)
        return TD

    def lets_init_weights(self):
        self.policy.lets_init_weights()
        self.policy_old.lets_init_weights()

    def save_weights(self):
        torch.save(self.policy.state_dict(), os.path.join(self.args.save_dir,'actor_ppo.pth'))
        torch.save(self.policy_old.state_dict(), os.path.join(self.args.save_dir,'old_actor_ppo.pth'))

    def load_weights(self):
        self.policy.load_state_dict(torch.load(os.path.join(self.args.save_dir,'actor_ppo.pth')))
        self.policy_old.load_state_dict(torch.load(os.path.join(self.args.save_dir,'old_actor_ppo.pth')))


if __name__ == "__main__":
    args = get_args()
    wandb.init(name=f"{args.test_id}_{args.ppo_episodes}", config=args)
    agent = Agent(args)
    wandb.watch(agent.policy, log_freq=100)
    i_ep = 0

    while i_ep < args.ppo_episodes:
        if 'saved_buffer.pkl' in os.listdir():
            not_ready = True
            print(f"Updating after Episode: {i_ep}")

            while not_ready:
                try:
                    agent.memory.loadBuffer()
                    agent.load_weights()
                except Exception as e:
                    not_ready = True
                else:
                    not_ready = False

            agent.update_ppo()
            agent.save_weights()
            i_ep += 1
