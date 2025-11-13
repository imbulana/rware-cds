# CDS Agent

import torch
import torch.nn.functional as F
from einops import rearrange

from .networks import CDSNetwork


class CDSAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        agent_specific_dim: int = 64,
        agent_specific_hidden: int = 64,
        device: str = 'cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = device
        
        self.network = CDSNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            agent_specific_dim=agent_specific_dim,
            agent_specific_hidden=agent_specific_hidden,
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
    
    def get_actions(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
        explore: bool = True,
    ):
        self.network.eval()
        with torch.no_grad():
            values, logits = self.network(obs, agent_ids)
            
            if explore:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            else:
                actions = logits.argmax(dim=-1)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
        
        return actions, log_probs, values
    
    def get_values(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
    ):
        self.network.eval()
        with torch.no_grad():
            values, _ = self.network(obs, agent_ids)
        return values
    
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_ids: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        l1_coef: float = 0.01,
        mi_coef: float = 0.1,
    ):
        self.network.train()
        
        # forward pass

        values, logits = self.network(obs, agent_ids)
        
        # policy loss (PPO-style)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO clipped objective

        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()
        
        # Value loss

        if values.dim() == 2 and returns.dim() == 2:
            value_loss = F.mse_loss(values, returns)
        else:
            value_loss = F.mse_loss(values.flatten(), returns.flatten())
        
        # L1 regularization on agent-specific parameters

        agent_specific_params = self.network.get_agent_specific_params()
        l1_loss = l1_coef * torch.norm(agent_specific_params, p=1)
        
        # mutual info regularization
        # (compute variance of agent-specific features across agents
        # higher variance => more diversity => higher mutual information

        mi_loss = self._compute_mi_loss(obs, agent_ids)
        
        # total loss

        total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy
            + l1_loss
            - mi_coef * mi_loss  # -ve b/c we want to maximize MI
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'l1_loss': l1_loss,
            'mi_loss': mi_loss,
        }
    
    def _compute_mi_loss(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
    ):
        # get shared features

        obs_flat = rearrange(obs, 'b n o -> (b n) o')
        agent_ids_flat = rearrange(agent_ids, 'b n -> (b n)')
        shared_features = self.network.shared_network.shared_encoder(obs_flat)
        
        # get agent-specific features for each agent

        agent_features_list = []
        for agent_id in range(self.n_agents):
            agent_mask = (agent_ids_flat == agent_id)
            if agent_mask.any():
                agent_shared = shared_features[agent_mask]
                agent_specific = self.network.agent_specific_modules[agent_id](agent_shared)
                agent_features_list.append(agent_specific)
        
        if len(agent_features_list) < 2:
            return torch.tensor(0.0, device=obs.device, requires_grad=True)
        
        # stack features: (n_agents, batch_per_agent, feature_dim)
        # want to maximize variance across agents
        # compute mean feature per agent

        agent_means = torch.stack([feat.mean(dim=0) for feat in agent_features_list])
        
        # variance across agents (higher => more diversity)
        # use variance as the metric (maximizing variance => maximizing MI)

        feature_variance = agent_means.var(dim=0).mean()
        
        return feature_variance
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        agent_ids: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        **loss_kwargs,
    ):
        # update the agent's network

        loss_dict = self.compute_loss(
            obs, actions, agent_ids, returns, old_log_probs, advantages, **loss_kwargs
        )
        
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
    
    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])