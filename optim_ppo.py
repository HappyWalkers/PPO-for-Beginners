import argparse
from part4.ppo_for_beginners.ppo_optimized import PPO
from part4.ppo_for_beginners.network import FeedForwardNN
from dm_control import suite,viewer
import numpy as np
from dmcontrol_to_gym_wrapper import DmcontrolToGymWrapper
import torch
import sys

def main(args: argparse.Namespace):
    hyperparameters = {
        'timesteps_per_batch': 256, 
        'max_timesteps_per_episode': 1000, 
        'gamma': 0.99, 
        'n_updates_per_iteration': 4,
        'lr': 5e-3, 
        'lam': 0.98,
        'clip': 0.2, 
        'save_freq': 500, 
        'seed': args.seed,
        'max_grad_norm':0.5,
        'target_kl':0.02,
        'ent_coef':0,
        'num_minibatches':6
    }
    total_timesteps = 405000

    r0 = np.random.RandomState(42)
    env = suite.load('walker', 'walk',
                    task_kwargs={'random': r0})
    env = DmcontrolToGymWrapper(env)

    if args.train:
        model = PPO(FeedForwardNN, env, **hyperparameters)
        
        # Tries to load in an existing actor/critic model to continue training on
        actor_model, critic_model = args.actor_model, args.critic_model
        if actor_model != '' and critic_model != '':
            print(f"Loading in {actor_model} and {critic_model}...", flush=True)
            model.actor.load_state_dict(torch.load(actor_model))
            model.critic.load_state_dict(torch.load(critic_model))
            print(f"Successfully loaded.", flush=True)
        elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
            print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
            sys.exit(0)
        else:
            print(f"Training from scratch.", flush=True)
            
        model.learn(total_timesteps)
        
    # if args.test:
    #     pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Train an agent using PPO.'))
    parser.add_argument('--train', action='store_true', help='Train the agent.', default=True)
    # parser.add_argument('--test', action='store_false', help='Test the agent.', default=False)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')
    args = parser.parse_args()
    main(args)