import torch
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from torch import nn

class PPOAgent():
    def __init__(self, observation_space, action_dim, 
                 learning_rate = 3e-4, discount = 0.99, 
                 epsilon = 0.2, entropy_coeff = 0.01):
        self.actor = ActorNetwork(observation_space, action_dim)
        self.critic = CriticNetwork(observation_space["grid"])
        self.learning_rate = learning_rate
        self.critic_loss_fn = nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.discount = discount
        self.epsilon = epsilon
        self.entropy_coeff = entropy_coeff
        
    def compute_rtgs(self, rewards, dones):
        rtgs = torch.zeros_like(rewards)

        rtg = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                rtg = 0.0
            rtg = rewards[t] + self.discount * rtg
            rtgs[t] = rtg

        return rtgs        

    def compute_advantages(self, rtgs, values):
        A = rtgs - values
        return (A - A.mean()) / (A.std() + 1e-8)
    
    def compute_loss(self, all_values, states, actions, action_log_probs, advantages, rtgs, masks):
        # critic (centralized)
        critic_loss = self.critic_loss_fn(rtgs, all_values)
        
        # actor
        logits = self.actor(states)
        logits = logits.masked_fill(masks == 0, -torch.inf)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions) # select the log_prob of the actions actually taken
        prob_ratios = torch.exp(new_log_probs-action_log_probs)
        surr_1 = prob_ratios * advantages
        surr_2 = torch.clamp(prob_ratios, 1-self.epsilon, 1+self.epsilon) * advantages
        entropy = dist.entropy().mean()
        actor_loss = -torch.mean(torch.min(surr_1, surr_2)) - self.entropy_coeff * entropy
        
        return critic_loss, actor_loss
    
    def optimize(self, critic_loss, actor_loss):
        # optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def act(self, obs, all_obs, masks):
        with torch.no_grad():
            logits = self.actor(obs)
            logits = logits.masked_fill(masks == 0, -torch.inf)
            dists = torch.distributions.Categorical(logits=logits)
            actions = dists.sample()
            log_probs = dists.log_prob(actions)
            values = self.critic(all_obs).squeeze(-1)
        return actions, log_probs, values

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))

    def set_actor_eval(self):
        self.actor.eval()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)