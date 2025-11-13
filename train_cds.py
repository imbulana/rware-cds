# training script for CDS algo on RWARE environment

import argparse
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime

from cds.agent import CDSAgent
from cds.trainer import CDSTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train CDS on RWARE environment')
    parser.add_argument('--env', type=str, default='rware:rware-tiny-2ag-v2',
                       help='Environment name')
    parser.add_argument('--n_steps', type=int, default=128,
                       help='Number of steps per rollout')
    parser.add_argument('--n_epochs', type=int, default=4,
                       help='Number of training epochs per rollout')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--n_iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension for shared network')
    parser.add_argument('--agent_specific_dim', type=int, default=64,
                       help='Dimension of agent-specific features')
    parser.add_argument('--agent_specific_hidden', type=int, default=64,
                       help='Hidden dimension for agent-specific modules')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy regularization coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                       help='Value loss coefficient')
    parser.add_argument('--l1_coef', type=float, default=0.01,
                       help='L1 regularization coefficient')
    parser.add_argument('--mi_coef', type=float, default=0.1,
                       help='Mutual information regularization coefficient')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save checkpoint every N iterations')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log metrics every N iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def get_env_info(env):
    # get observation space

    if hasattr(env, 'observation_space'):
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_dim = env.observation_space.shape[0]
        elif isinstance(env.observation_space, gym.spaces.Tuple):
            # Assume all agents have same observation space
            obs_dim = env.observation_space.spaces[0].shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")
    else:
        raise ValueError("Environment does not have observation_space attribute")
    
    # get action space

    if hasattr(env, 'action_space'):
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action_dim = env.action_space.nvec[0]  # Assume all agents have same action space
        else:
            raise ValueError(f"Unsupported action space: {env.action_space}")
    else:
        raise ValueError("Environment does not have action_space attribute")
    
    # get number of agents

    if hasattr(env, 'n_agents'):
        n_agents = env.n_agents
    elif hasattr(env, 'num_agents'):
        n_agents = env.num_agents
    elif isinstance(env.observation_space, gym.spaces.Tuple):
        n_agents = len(env.observation_space.spaces)
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        n_agents = len(env.action_space.nvec)
    else:
        n_agents = 1
    
    return obs_dim, action_dim, n_agents


def main():
    args = parse_args()
    
    # set random seeds

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # create environment

    env = gym.make(args.env)
    obs, info = env.reset(seed=args.seed)
    
    # get environment info

    obs_dim, action_dim, n_agents = get_env_info(env)
    
    print(f"Environment: {args.env}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Number of agents: {n_agents}")
    
    if isinstance(obs, tuple):
        obs = np.stack(obs)
    
    # create agent

    agent = CDSAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=args.hidden_dim,
        agent_specific_dim=args.agent_specific_dim,
        agent_specific_hidden=args.agent_specific_hidden,
        device=args.device,
    )
    
    # update lr

    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = args.lr
    
    # create trainer

    trainer = CDSTrainer(
        env=env,
        agent=agent,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # create save directory

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # training loop

    print("\nStarting training...")
    metrics_history = []
    
    for iteration in range(args.n_iterations):
        metrics, obs = trainer.train_step(
            obs,
            loss_kwargs={
                'entropy_coef': args.entropy_coef,
                'value_coef': args.value_coef,
                'l1_coef': args.l1_coef,
                'mi_coef': args.mi_coef,
            }
        )
        
        metrics['iteration'] = iteration
        metrics_history.append(metrics)
        
        # logging

        if (iteration + 1) % args.log_interval == 0:
            avg_metrics = {
                k: np.mean([m[k] for m in metrics_history[-args.log_interval:]]) 
                for k in metrics.keys() if k != 'iteration'
            }

            print(f"\nIteration {iteration + 1}/{args.n_iterations}")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        # save checkpoint

        if (iteration + 1) % args.save_interval == 0:
            checkpoint_path = save_dir / f"checkpoint_{iteration + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            agent.save(str(checkpoint_path))
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    # save final checkpoint and metrics

    final_checkpoint_path = save_dir / f"checkpoint_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    agent.save(str(final_checkpoint_path))
    
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"\nTraining complete.")
    print(f"Final checkpoint saved to {final_checkpoint_path}")
    print(f"Metrics saved to {metrics_path}")
    
    env.close()


if __name__ == '__main__':
    main()