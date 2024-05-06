import argparse
from part4.ppo_for_beginners.ppo_optimized import PPO
from part4.ppo_for_beginners.network import FeedForwardNN
from dm_control import suite,viewer
import numpy as np
from dmcontrol_to_gym_wrapper import DmcontrolToGymWrapper

def main(args: argparse.Namespace):
    hyperparameters = {
        'timesteps_per_batch': 256, 
        'max_timesteps_per_episode': 1000, 
        'gamma': 0.99, 
        'n_updates_per_iteration': 4,
        'lr': 5e-3, 
        'lam': 0.98,
        'clip': 0.2, 
        'save_freq': 1e15, 
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
        model.learn(total_timesteps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Train an agent using PPO.'))
    parser.add_argument('--train', action='store_true', help='Train the agent.', default=True)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    args = parser.parse_args()
    main(args)