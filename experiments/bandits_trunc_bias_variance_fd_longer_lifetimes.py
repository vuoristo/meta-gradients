import itertools

import numpy as np
from ray import tune

from algorithms.off_policy_bandits import trainable


if __name__ == '__main__':
    side = np.linspace(0.5, 2.5, 5)
    grid = list(itertools.product(side, side))
    config = dict(
        interact_batch_size=10,
        replay_batch_size=10,
        meta_batch_size=10,
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
        truncation_length=80,
        meta_discount=1.0,
        logpi_cum_fn='meta_discount',
        e_maml_lambda=1.0,
        opt_type='sgd',
        opt_kwargs=dict(learning_rate=1e-1),
        inner_loss=tune.grid_search(['no_is']),
        outer_loss='outer_loss_ignore_buffer',
        separate_buffers=False,
        off_policy_inner=False,
        off_policy_outer=False,
        seed=tune.grid_search(list(range(5))),
        epsilon=1e-2,
        iterations=200,
        parallel_runs=32000,
        compute_fd=True,
        compute_pg=False,
        update_with_fd=False,
        update_with_pg=False,
    )

    analysis = tune.run(
        trainable,
        name='on_policy_bandits:learn_lr_schedule:lt_len_80:noise_2:fd:range_05_25:index_cutoff_10:epsilon_001:more_samples',
        config=config,
        verbose=1,
        resources_per_trial={
            'gpu': 1.0,
        },
    )
