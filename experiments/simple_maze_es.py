import itertools

import numpy as np
from ray import tune

from algorithms.learn_entropy_es_multilife import trainable


if __name__ == '__main__':
    config = {
        'env_type': 'simple_maze',
        'env_kwargs': {
            'side': 5,
            'episodic': True,
            'fixed_length_episodes': True,
            'episode_max_len': 16,
            'reward_noise_std': 0.0,
        },
        'num_actions': 4,
        'reward_flip_interval': 6_400,
        'randomize_lt': False,
        'reset_theta_at_reward_flip': False,
        'H': 16,
        'inner_batch_size': tune.grid_search([5]),
        'outer_batch_size': 50,
        'gamma': 0.99,

        'inner_loop_fn': 'default',
        'inner_loss': 'no_is',
        'outer_loss': 'first_state',
        'meta_lambda': 0.0,
        'truncation_length': tune.grid_search([8, 16, 32, 64, 200, 400]),
        'auc_loss': True,
        'eta_context_len': 10,
        'vf_coef': 0.1,
        'ent_coef': 0.0,

        'train_step': 'es',
        'fd_epsilon': 0.1,
        'save_gradients': False,

        'theta_net': 'actor_critic_net',

        'eta_net': 'context',
        'eta_net_kwargs': {},

        'inner_weight_decay': 0.0,
        'inner_opt_type': 'sgd',
        'inner_opt_kwargs': {
            'learning_rate': 1.0,
        },
        'max_inner_grad_norm': 0.0,

        # ES
        'es_strategy': 'OpenES',
        'es_params': {
            'lrate_init': tune.grid_search([1e-6, 5e-6]),
            'beta_1': 0.9,
            'beta_2': 0.999,
            'eps': 1e-4,
            'lrate_decay': 1.0,
        },

        'log_interval_steps': 1_024_000,
        'stop_steps': 20_000_000_000,
        'eval_interval_steps': 1_000_000_000,
        'eval_steps': 100_000_000,
        'seed': tune.grid_search([0, 1, 2]),
    }
    analysis = tune.run(
        trainable,
        name='simple_maze_5:flip_interval_6400:no_reset_theta:es:openes:ilr_1:trunc_8_16_32_64_200_400:olr_sweep',
        config=config,
        resources_per_trial={
            'gpu': 0.3,
        },
    )

