import itertools

import numpy as np
from ray import tune

from new_tech.off_policy_bandits import trainable


if __name__ == '__main__':
    side = np.linspace(0.5, 2.5, 5)
    grid = list(itertools.product(side, side))
    config = dict(
        interact_batch_size=10,
        replay_batch_size=10,
        meta_batch_size=10,
        popsize=2,
        buffer_size=tune.sample_from(
            lambda spec: spec.get('config', spec)['lifetime_length']
                         * spec.get('config', spec)['interact_batch_size']),
        initial_eta=tune.grid_search(grid),
        inv_temp_fn_type='constant',
        learning_rate_fn_type='learned',
        K=30,
        reward_noise_std=2.0,
        inner_learning_rate=0.0,
        lifetime_length=80,
        index_cutoff_ratio=10,
        truncation_length=tune.grid_search([80]),
        meta_discount=1.0,
        logpi_cum_fn='meta_batch_size',
        e_maml_lambda=tune.grid_search([0.0, 2.0, 4.0, 6.0, 8.0]),
        opt_type='sgd',
        opt_kwargs=dict(learning_rate=1e-1),
        inner_loss=tune.grid_search(['no_is']),
        outer_loss='outer_loss_ignore_buffer',
        separate_buffers=False,
        off_policy_inner=False,
        off_policy_outer=False,
        seed=tune.grid_search(list(range(5))),
        epsilon=1e-1,
        iterations=200,
        parallel_runs=4000,
        compute_fd=False,
        compute_pg=True,
        update_with_fd=False,
        update_with_pg=False,
    )

    analysis = tune.run(
        trainable,
        name='on_policy_bandits:learn_lr_schedule:lt_len_80:noise_2:mg:range_05_25:index_cutoff_10:ml_0to8',
        config=config,
        verbose=1,
        resources_per_trial={
            'gpu': 1.0,
        },
    )
