# Nets for CDS

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class AgentSpecificModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor):
        return self.network(x)


class SharedNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        agent_specific_dim: int = 64,
        use_agent_specific: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.agent_specific_dim = agent_specific_dim
        self.use_agent_specific = use_agent_specific
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        agent_input_dim = hidden_dim
        
        self.value_head = nn.Linear(hidden_dim + (agent_specific_dim if use_agent_specific else 0), 1)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + (agent_specific_dim if use_agent_specific else 0), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        agent_specific_features: torch.Tensor = None,
    ):

        # extract shared features

        shared_features = self.shared_encoder(obs)
        
        # combine with agent-specific features if available

        if self.use_agent_specific and agent_specific_features is not None:
            combined_features = torch.cat([shared_features, agent_specific_features], dim=-1)
        else:
            combined_features = shared_features
        
        # compute value and policy

        value = self.value_head(combined_features).squeeze(-1)
        logits = self.policy_head(combined_features)
        
        return value, logits


class CDSNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        agent_specific_dim: int = 64,
        agent_specific_hidden: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.agent_specific_dim = agent_specific_dim
        
        self.shared_network = SharedNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            agent_specific_dim=agent_specific_dim,
            use_agent_specific=True,
        )
        
        self.agent_specific_modules = nn.ModuleList([
            AgentSpecificModule(
                input_dim=hidden_dim,
                hidden_dim=agent_specific_hidden,
                output_dim=agent_specific_dim,
            )
            for _ in range(n_agents)
        ])
    
    def forward(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
    ):

        # handle different input shapes

        original_shape = obs.shape
        if len(original_shape) == 3:
            batch_size, n_agents, obs_dim = original_shape
            seq_len = None
            obs_flat = rearrange(obs, 'b n o -> (b n) o')
            agent_ids_flat = rearrange(agent_ids, 'b n -> (b n)')
        elif len(original_shape) == 4:
            # (batch, seq_len, n_agents, obs_dim)
            batch_size, seq_len, n_agents, obs_dim = original_shape
            obs_flat = rearrange(obs, 'b t n o -> (b t n) o')
            agent_ids_flat = rearrange(agent_ids, 'b t n -> (b t n)')
        else:
            raise ValueError(f"Unexpected observation shape: {original_shape}")
        
        # extract shared and agent-specific features

        shared_features = self.shared_network.shared_encoder(obs_flat)
        
        agent_specific_features_list = []
        for agent_id in range(self.n_agents):
            agent_mask = (agent_ids_flat == agent_id)
            if agent_mask.any():
                agent_obs = shared_features[agent_mask]
                agent_features = self.agent_specific_modules[agent_id](agent_obs)
                agent_specific_features_list.append((agent_mask, agent_features))
        
        # combine agent-specific features

        agent_specific_features = torch.zeros(
            obs_flat.shape[0],
            self.agent_specific_dim,
            device=obs_flat.device,
            dtype=obs_flat.dtype,
        )
        for agent_mask, features in agent_specific_features_list:
            agent_specific_features[agent_mask] = features
        
        # compute value and logits

        value, logits = self.shared_network(obs_flat, agent_specific_features)
        
        # reshape back to original structure

        if seq_len is None:
            # (batch, n_agents)
            values = rearrange(value, '(b n) -> b n', b=batch_size, n=n_agents)
            # (batch, n_agents, action_dim)
            logits = rearrange(logits, '(b n) a -> b n a', b=batch_size, n=n_agents, a=self.action_dim)
        else:
            # (batch, seq_len, n_agents)
            values = rearrange(value, '(b t n) -> b t n', b=batch_size, t=seq_len, n=n_agents)
            # (batch, seq_len, n_agents, action_dim)
            logits = rearrange(
                logits, '(b t n) a -> b t n a', b=batch_size, t=seq_len, n=n_agents, a=self.action_dim
            )
        
        return values, logits
    
    def get_agent_specific_params(self) -> torch.Tensor:
        params = []
        for module in self.agent_specific_modules:
            params.append(torch.cat([p.flatten() for p in module.parameters()]))
        return torch.cat(params) if params else torch.tensor([], device=next(self.parameters()).device)