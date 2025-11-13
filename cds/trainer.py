# Trainer for CDS

import torch
import numpy as np
from einops import rearrange
import gymnasium as gym

from .agent import CDSAgent


class CDSTrainer:
    def __init__(
        self,
        env: gym.Env,
        agent: CDSAgent,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = 'cpu',
    ):
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        
        self.n_agents = getattr(env, 'n_agents', env.num_agents if hasattr(env, 'num_agents') else 1)
        
        if hasattr(env, 'n_agents'):
            self.n_agents = env.n_agents
        elif hasattr(env, 'num_agents'):
            self.n_agents = env.num_agents
        else:
            if hasattr(env.action_space, 'n'):
                self.n_agents = 1
            elif hasattr(env.action_space, 'spaces'):
                self.n_agents = len(env.action_space.spaces)
            else:
                raise ValueError("Cannot determine number of agents from environment")
    
    def collect_rollout(
        self,
        obs: np.ndarray,
    ):
        # collect a rollout of experiences

        obs_list = []
        action_list = []
        reward_list = []
        value_list = []
        log_prob_list = []
        done_list = []
        agent_id_list = []
        
        current_obs = obs
        done = False
        truncated = False
        
        for step in range(self.n_steps):
            if isinstance(current_obs, tuple):
                obs_array = np.stack(current_obs)
            else:
                obs_array = current_obs
            
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)  # (1, n_agents, obs_dim)
            
            # create agent IDs

            agent_ids = torch.arange(self.n_agents, device=self.device).unsqueeze(0)  # (1, n_agents)
            
            # get actions

            actions, log_probs, values = self.agent.get_actions(obs_tensor, agent_ids, explore=True)
            
            # step environment

            if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                actions_np = actions.cpu().numpy().squeeze(0)
            elif isinstance(self.env.action_space, gym.spaces.Box):
                actions_np = actions.cpu().numpy().squeeze(0)
            else:
                actions_np = actions.cpu().numpy().squeeze(0)
            
            next_obs, reward, done, truncated, info = self.env.step(actions_np)
            
            # store experience

            obs_list.append(obs_array)
            action_list.append(actions.cpu().numpy().squeeze(0))
            reward_list.append(reward)
            value_list.append(values.cpu().numpy().squeeze(0))
            log_prob_list.append(log_probs.cpu().numpy().squeeze(0))
            done_list.append(done or truncated)
            agent_id_list.append(np.arange(self.n_agents))
            
            if done or truncated:
                # reset environment if done

                current_obs, _ = self.env.reset()
                if isinstance(current_obs, tuple):
                    current_obs = np.stack(current_obs)
                break
            else:
                current_obs = next_obs
        
        # get final value for bootstrapping

        if not (done or truncated):
            if isinstance(current_obs, tuple):
                final_obs = np.stack(current_obs)
            else:
                final_obs = current_obs
            final_obs_tensor = torch.FloatTensor(final_obs).unsqueeze(0).to(self.device)
            final_agent_ids = torch.arange(self.n_agents, device=self.device).unsqueeze(0)
            final_values = self.agent.get_values(final_obs_tensor, final_agent_ids)
            final_values_np = final_values.cpu().numpy().squeeze(0)
        else:
            final_values_np = np.zeros(self.n_agents)
        
        rollout = {
            'obs': np.array(obs_list),              # (n_steps, n_agents, obs_dim)
            'actions': np.array(action_list),       # (n_steps, n_agents)
            'rewards': np.array(reward_list),       # (n_steps,)
            'values': np.array(value_list),         # (n_steps, n_agents)
            'log_probs': np.array(log_prob_list),   # (n_steps, n_agents)
            'dones': np.array(done_list),           # (n_steps,)
            'agent_ids': np.array(agent_id_list),   # (n_steps, n_agents)
            'final_values': final_values_np,        # (n_agents,)
        }
        
        return rollout
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        final_values: np.ndarray,
    ):

        # compute generalized advantage estimation (GAE)

        n_steps = len(rewards)
        n_agents = values.shape[1]
        
        advantages = np.zeros((n_steps, n_agents))
        returns = np.zeros((n_steps, n_agents))
        
        last_gae = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_value = final_values
            else:
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            advantages[step] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * last_gae
            returns[step] = advantages[step] + values[step]
        
        return advantages, returns
    
    def train_step(
        self,
        obs: np.ndarray,
        loss_kwargs = None,
    ):
        # perform one training step

        if loss_kwargs is None:
            loss_kwargs = {}
        
        # collect rollout

        rollout = self.collect_rollout(obs)
        
        # compute advantages and returns

        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones'],
            rollout['final_values'],
        )
        
        # normalize advantages

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # convert to tensors

        obs_tensor = torch.FloatTensor(rollout['obs']).to(self.device)
        actions_tensor = torch.LongTensor(rollout['actions']).to(self.device)
        agent_ids_tensor = torch.LongTensor(rollout['agent_ids']).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(rollout['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # reshape for training: (n_steps, n_agents, ...) -> (n_steps * n_agents, ...)

        n_steps, n_agents = obs_tensor.shape[:2]
        obs_flat = rearrange(obs_tensor, 't n o -> (t n) o')
        actions_flat = rearrange(actions_tensor, 't n -> (t n)')
        agent_ids_flat = rearrange(agent_ids_tensor, 't n -> (t n)')
        returns_flat = rearrange(returns_tensor, 't n -> (t n)')
        old_log_probs_flat = rearrange(old_log_probs_tensor, 't n -> (t n)')
        advantages_flat = rearrange(advantages_tensor, 't n -> (t n)')
        
        # train for multiple epochs

        total_metrics = {}
        n_samples = len(obs_flat)
        
        for epoch in range(self.n_epochs):
            # shuffle data

            indices = np.random.permutation(n_samples)
            
            # mini-batch training

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs_flat[batch_indices]
                batch_actions = actions_flat[batch_indices]
                batch_agent_ids = agent_ids_flat[batch_indices]
                batch_returns = returns_flat[batch_indices]
                batch_old_log_probs = old_log_probs_flat[batch_indices]
                batch_advantages = advantages_flat[batch_indices]
                
                # reshape to (batch, 1, ...) for agent update
                # the agent expects (batch, n_agents, ...) format

                batch_obs_reshaped = batch_obs.unsqueeze(1)
                batch_actions_reshaped = batch_actions.unsqueeze(1)
                batch_agent_ids_reshaped = batch_agent_ids.unsqueeze(1)
                batch_returns_reshaped = batch_returns.unsqueeze(1)
                batch_old_log_probs_reshaped = batch_old_log_probs.unsqueeze(1)
                batch_advantages_reshaped = batch_advantages.unsqueeze(1)
                
                # update agent

                metrics = self.agent.update(
                    batch_obs_reshaped,
                    batch_actions_reshaped,
                    batch_agent_ids_reshaped,
                    batch_returns_reshaped,
                    batch_old_log_probs_reshaped,
                    batch_advantages_reshaped,
                    **loss_kwargs,
                )
                
                # accumulate metrics

                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = []
                    total_metrics[key].append(value)
        
        # average metrics

        avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
        
        # get next observation for next rollout
        # check if environment was done/truncated

        if rollout['dones'][-1] if len(rollout['dones']) > 0 else True:
            # reset environment if done

            next_obs, _ = self.env.reset()
            if isinstance(next_obs, tuple):
                next_obs = np.stack(next_obs)
        else:
            # Use the last observation from rollout
            next_obs = rollout['obs'][-1]
        
        return avg_metrics, next_obs