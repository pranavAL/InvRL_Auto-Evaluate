import os
import time
import wandb
import torch
import numpy as np
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
        self.action_std = 0.5
        self.gamma = args.gamma
        self.lam = args.lam
        self.args = args
        self.loss_epoch = 0

        self.policy = PPO(self.state_dim, self.action_dim, self.args).to(self.args.device)
        self.policy_old = PPO(self.state_dim, self.action_dim, self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.betas = (0.9, 0.999)

        self.policy_actor_optimizer = torch.optim.Adam(self.policy.actor_layer.parameters(), lr=self.lr_actor, betas=self.betas)
        self.policy_critic_optimizer = torch.optim.Adam(self.policy.critic_layer.parameters(), lr=self.lr_critic, betas=self.betas)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_actor_optimizer, step_size=5000, gamma=0.96)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_critic_optimizer, step_size=5000, gamma=0.96)

        self.memory = Memory()
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(self.args.device)
        self.cov_mat = torch.diag_embed(self.action_var).to(self.args.device).detach()

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

        vpredclipped = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1 = (returns - values).pow(2) # Mean Squared Error
        vf_losses2 = (vpredclipped - returns).pow(2) # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean() * 0.5

        logprobs = distribution.log_prob(actions).float().to(self.args.device)
        old_distribution = MultivariateNormal(old_action_mean, self.cov_mat)
        old_logprobs = old_distribution.log_prob(actions).float().to(self.args.device).detach()

        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        pg_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy.mean()

        self.meta_data['Advantage'].append(advantages.mean().item())
        self.meta_data['Entropy'].append(dist_entropy.mean().item())
        self.meta_data['TD'].append(returns.mean().item())
        self.meta_data['Critic_Loss'].append(critic_loss.item())
        self.meta_data['KL'].append(ratios.mean().item())
        self.meta_data['Policy_Loss'].append(pg_loss.item())

        return critic_loss, pg_loss

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

            critic_loss, pg_loss = self.evaluate_loss(states, actions, rewards, next_states, dones)
            self.policy_critic_optimizer.zero_grad()
            self.policy_actor_optimizer.zero_grad()

            critic_loss.backward()
            self.policy_critic_optimizer.step()

            pg_loss.backward()
            self.policy_actor_optimizer.step()
            self.critic_scheduler.step()
            self.actor_scheduler.step()

        self.memory.deleteBuffer()
        self.policy_old.load_state_dict(self.policy.state_dict())

        wandb.log({'Advantage' : np.mean(self.meta_data['Advantage'])})
        wandb.log({'Entropy' :np.mean(self.meta_data['Entropy'])})
        wandb.log({'TD': np.mean(self.meta_data['TD'])})
        wandb.log({'Critic_Loss': np.mean(self.meta_data['Critic_Loss'])})
        wandb.log({'KL': np.mean(self.meta_data['KL'])})
        wandb.log({'Policy_Loss': np.mean(self.meta_data['Policy_Loss'])})

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
    wandb.init(name="Excavator Learning Policy", config=args)
    agent = Agent(args)
    wandb.watch(agent.policy, log_freq=100)
    i_ep = 0

    while True:
        if 'saved_buffer.pkl' in os.listdir():
            time.sleep(10)
            print(f"Updating after Episode: {i_ep}")
            agent.memory.loadBuffer()
            agent.load_weights()
            agent.update_ppo()
            agent.save_weights()
            i_ep += 1
