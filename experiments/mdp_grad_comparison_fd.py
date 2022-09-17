import itertools

import numpy as np
from ray import tune

from algorithms.learn_entropy_mg_multilife import trainable



if __name__ == '__main__':
    config = {
        'env_type': 'simple_maze',
        'env_kwargs': {
            'side': 3,
            'episodic': True,
            'fixed_length_episodes': True,
            'episode_max_len': 8,
        },
        'num_actions': 4,
        'reward_flip_interval': tune.sample_from(
            lambda x: x.config['truncation_length']
                      * x.config['H']),
        'randomize_lt': False,
        'reset_theta_at_reward_flip': True,
        'H': 8,
        'inner_batch_size': 10,
        'outer_batch_size': 25,
        'inner_learning_rate': 1.0,
        'gamma': 0.99,

        'inner_loop_fn': tune.grid_search(['default']),
        'inner_loss': 'no_is',
        'outer_loss': tune.grid_search(['first_state_adv']),
        'truncation_length': 8,
        'meta_lambda': 1.0,
        'auc_loss': True,
        'vf_coef': 0.1,
        'ent_coef': 0.0,

        'train_step': tune.grid_search(['fd']),
        'fd_epsilon': 0.1,
        'save_gradients': True,

        'theta_net': 'actor_critic_net',

        'eta_context_len': 10,
        'eta_net': 'simple',
        'eta_net_kwargs': {
            'initial_eta': tune.grid_search(np.linspace(-2.0, 4.0, num=10).tolist()),
        },

        'inner_weight_decay': 0.0,
        'inner_opt_type': 'sgd',
        'inner_opt_kwargs': {
            'learning_rate': 1.0,
        },
        'max_inner_grad_norm': 0.0,

        'outer_weight_decay': 0.0,
        'opt_type': 'adam',
        'opt_kwargs': {
            'learning_rate': 0.0,
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1E-4,
        },
        'max_outer_grad_norm': 0.0,
        'log_interval_steps': 100_000,
        'stop_steps': 50_000_000,
        'eval_interval_steps': 100_000_000,
        'eval_steps': 5_000_000,
        'seed': tune.grid_search(list(range(20))),
    }
    analysis = tune.run(
        trainable,
        name='simple_maze_3:reset_theta:trunc_8:vf_coef_01:sweep_outer_loss:fd_only',
        config=config,
        resources_per_trial={
            'gpu': 0.5,
        },
    )
