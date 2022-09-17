import itertools

import numpy as np
from ray import tune

from algorithms.off_policy_bandits import trainable


if __name__ == '__main__':
    config = dict(
        interact_batch_size=10,
        replay_batch_size=10,
        meta_batch_size=10,
        buffer_size=tune.sample_from(
            lambda spec: spec.get('config', spec)['lifetime_length']
                         * spec.get('config', spec)['interact_batch_size']),
        initial_eta=(2.0, 2.0),
        inv_temp_fn_type='constant',
        learning_rate_fn_type='learned',
        K=30,
        reward_noise_std=2.0,
        inner_learning_rate=0.0,
        lifetime_length=30,
        index_cutoff_ratio=4,
        truncation_length=tune.grid_search([2, 9]),
        meta_discount=1.0,
        logpi_cum_fn='meta_batch_size',
        e_maml_lambda=tune.grid_search([0.0, 2.0, 6.0, 10.0]),
        opt_type='sgd',
        opt_kwargs=dict(learning_rate=0.05),
        inner_loss=tune.grid_search(['no_is']),
        outer_loss='outer_loss_ignore_buffer',
        separate_buffers=False,
        off_policy_inner=False,
        off_policy_outer=False,
        seed=tune.grid_search(list(range(5))),
        epsilon=1e-1,
        iterations=500000,
        parallel_runs=1000,
        compute_fd=False,
        compute_pg=True,
        update_with_fd=False,
        update_with_pg=True,
    )

    analysis = tune.run(
        trainable,
        name='on_policy_bandits:learn_lr_schedule:lt_len_30:noise_2:eta_init_22:lr_005:500k_steps:trunc_in_2_9:sweep_elambda',
        config=config,
        verbose=1,
        resources_per_trial={
            'gpu': 0.2,
        },
    )
