import os
import functools
import collections
from typing import Dict, Tuple, Callable, Union, Any
import pickle
import glob

import jax
import jax.numpy as jnp
import jax.random as jrandom
import rlax
import chex
import optax
import pandas as pd
import haiku as hk
from ray import tune
import ray

from algorithms.utils import pack_namedtuple_jnp
from environments.simple_maze import maze_env
from algorithms.learn_entropy_mg_multilife import (
    Log,
    TimeStep,
    ActorCriticNet,
    ActorNet,
    get_eta_net,
    inner_unroll,
    get_train_step_fn,
)


Array = chex.Array
Scalar = chex.Scalar
Numeric = chex.Numeric


def log_status(
    reporter: Any,
    rewards: Array,
    total_steps: Scalar,
    total_reward: Scalar,
    inner_log: Log,
    outer_log: Log,
):
    outer_log = jax.device_get(outer_log)
    inner_log = jax.device_get(inner_log)
    for t in range(rewards.shape[1]):
        l = dict(
            reward_per_timestep=rewards[:, t].mean(),
            entropy=inner_log.entropy[:, t].mean(),
            pi_loss=inner_log.pi_loss[:, t].mean(),
            baseline_loss=inner_log.baseline_loss[:, t].mean(),
            inner_grad_norm=inner_log.grad_norm[:, t].mean(),
            ent_coef=inner_log.ent_coef[:, t].mean(),
            total_steps=total_steps,
            total_reward=total_reward,
        )
        reporter(**l)


def trainable(
    config: Dict,
    reporter: Any,
):
    num_trajs = config['inner_batch_size']
    num_envs = config['outer_batch_size']
    num_actions = config['num_actions']
    if config['theta_net'] == 'actor_critic_net':
        init_theta, apply_theta = hk.without_apply_rng(
            hk.transform(lambda inputs: ActorCriticNet(num_actions=num_actions, )(inputs)))
    elif config['theta_net'] == 'actor_net':
        init_theta, apply_theta = hk.without_apply_rng(
            hk.transform(lambda inputs: ActorNet(num_actions=num_actions, )(inputs)))
    eta_net_class = get_eta_net(config)
    init_eta, apply_eta = hk.without_apply_rng(
        hk.transform(lambda inputs, state: eta_net_class(**config['eta_net_kwargs'])(inputs, state)))
    _, init_eta_state_fn = hk.without_apply_rng(
        hk.transform(lambda batch_size: eta_net_class(
            **config['eta_net_kwargs']).get_initial_state(batch_size)))
    if config['env_type'] == 'simple_maze':
        (reset_env_fn, step_env_fn, resample_goals_fn
            ) = maze_env.get_maze_env(**config['env_kwargs'])
    if config['inner_opt_type'] == 'sgd':
        inner_opt_kwargs = config['inner_opt_kwargs'].copy()
        learning_rate = inner_opt_kwargs.pop('learning_rate')
        inner_optimizer = optax.chain(
            optax.inject_hyperparams(optax.sgd)(learning_rate=learning_rate),
        )
    if config['max_inner_grad_norm'] > 0:
        inner_optimizer = optax.chain(
            optax.clip_by_global_norm(config['max_inner_grad_norm']),
            inner_optimizer,
        )
    inner_unroll_ = functools.partial(
        inner_unroll,
        step_env_fn=step_env_fn,
        reset_env_fn=reset_env_fn,
        apply_theta_fn=apply_theta,
        apply_eta_fn=apply_eta,
        init_theta_fn=init_theta,
        resample_goals_fn=resample_goals_fn,
        inner_optimizer=inner_optimizer,
        init_eta_state_fn=init_eta_state_fn,
        config=config,
    )
    if config['opt_type'] == 'adam':
        opt_kwargs = config['opt_kwargs'].copy()
        learning_rate = opt_kwargs.pop('learning_rate')
        optimizer = optax.chain(
            optax.additive_weight_decay(config['outer_weight_decay']),
            optax.scale_by_adam(**opt_kwargs),
            optax.scale(-learning_rate),
        )
    elif config['opt_type'] == 'sgd':
        opt_kwargs = config['opt_kwargs'].copy()
        learning_rate = opt_kwargs.pop('learning_rate')
        optimizer = optax.sgd(learning_rate=learning_rate)
    if config['max_outer_grad_norm'] > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['max_outer_grad_norm']),
            optimizer,
        )
    train_step_ = jax.jit(functools.partial(
        get_train_step_fn(config),
        inner_loop_fn=inner_unroll_,
        eta_optimizer=optimizer,
    ))
    k1 = jrandom.PRNGKey(config['seed'])
    k1, k2 = jrandom.split(k1)
    keys = jrandom.split(k2, num_trajs * num_envs)
    obs = jax.tree_map(lambda x: x.reshape(num_envs, num_trajs, *x.shape[1:]),
                       jax.vmap(reset_env_fn)(keys))
    k1, k2 = jrandom.split(k1)
    keys = jrandom.split(k2, num_envs)
    goals = jax.vmap(resample_goals_fn)(keys)
    goals = goals[:, None].tile((1, num_trajs, *(1, ) * (goals.ndim - 1)))
    if config['randomize_lt']:
        k1, k2 = jrandom.split(k1)
        trunc_window_len = config['truncation_length'] * config['env_kwargs']['episode_max_len']
        ltmax = config['reward_flip_interval'] // trunc_window_len
        ltl = jrandom.randint(k2, (num_envs,), minval=0, maxval=ltmax)
        ltl = ltl[:, None].tile((1, num_trajs)) * trunc_window_len
    else:
        ltl = jnp.zeros((num_envs, num_trajs, ))
    ts = TimeStep(
        action_tm1=jnp.zeros((num_envs, num_trajs, ), dtype=jnp.int32),
        reward=jnp.zeros((num_envs, num_trajs, )),
        discount=jnp.zeros((num_envs, num_trajs, )),
        observation=obs,
        episode_length=jnp.zeros((num_envs, num_trajs, )),
        lifetime_length=ltl,
        goals=goals,
    )
    k1, k2 = jrandom.split(k1)
    keys = jrandom.split(k2, num_envs)
    theta = jax.vmap(init_theta)(keys, obs)
    mean_rewards = jnp.zeros((num_envs, config['eta_context_len']))
    dummy_state = init_eta_state_fn(None, 1)
    k1, k2 = jrandom.split(k1)
    if config['eta_net'] == 'frodo':
        eta_inputs = dict(
            reward=jnp.zeros_like(ts.reward[:, :1]),
            discount=jnp.zeros_like(ts.reward[:, :1]),
            pi_a=jnp.zeros_like(ts.reward[:, :1]),
            mu_a=jnp.zeros_like(ts.reward[:, :1]),
            v_t=jnp.zeros_like(ts.reward[:, :1]),
        )
        eta = init_eta(k2, eta_inputs, dummy_state)
    else:
        eta = init_eta(k2, mean_rewards, dummy_state)
    eta_opt_state = optimizer.init(eta)
    with open(config['checkpoint_path'], 'rb') as f:
        theta, eta, eta_opt_state = pickle.load(f)
    k1, k2 = jrandom.split(k1)
    total_reward = 0.0
    total_steps = 0
    t = 0
    while total_steps < config['stop_steps']:
        k1, k2 = jrandom.split(k1)
        outputs = train_step_(k2, eta, theta, mean_rewards, ts, eta_opt_state)
        (eta, theta, rewards, mean_rewards, ts, eta_opt_state,
            outer_log, inner_log, g, theta_nan, eta_nan, old_theta, old_eta,
            old_eta_opt_state) = outputs
        total_reward += rewards.sum()
        total_steps += rewards.size
        t += 1
        log_status(
            reporter=reporter,
            rewards=rewards,
            inner_log=inner_log,
            outer_log=outer_log,
            total_steps=total_steps,
            total_reward=total_reward,
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ray')
    parser.add_argument('--trial_path', required=True)
    args = parser.parse_args()

    analysis = tune.Analysis(args.trial_path)
    configs = analysis.get_all_configs()
    dfs = analysis.trial_dataframes
    all_ckpts = []
    all_configs = collections.defaultdict(list)
    for key, df in dfs.items():
        checkpoints = glob.glob(os.path.join(key, '*.ckpt'))
        for ckpt in checkpoints:
            all_ckpts.append(ckpt)
            for k, v in configs[key].items():
                all_configs[k].append(v)

    eval_config = {
        'key': tune.grid_search(list(range(len(all_ckpts))))
    }
    # Create lambda inside a function to create a reference to
    # the correct values within the lambda
    def get_getter(values):
        return lambda spec: values[spec.get('config', spec)['key']]
    for k, v in all_configs.items():
        eval_config[k] = tune.sample_from(get_getter(v))
    eval_config['stop_steps'] = 100_000_000
    eval_config.update({
        'checkpoint_path': tune.sample_from(get_getter(all_ckpts)),
        'stop_steps': 50_000_000,
    })
    if args.mode == 'ray':
        analysis = tune.run(
            trainable,
            name='evaluate_simple_maze_5:trunc_8_16_32:flip_interval_25600:reset_theta:inner_lr_01:ibs_5:obs_50:sweep_lr',
            config=eval_config,
            resources_per_trial={
                'gpu': 1.0,
            },
        )
    elif args.mode == 'ray_single':
        _, config = next(tune.suggest.variant_generator.generate_variants(eval_config))
        analysis = tune.run(
            trainable,
            name='20220114_test_eval:2',
            config=config,
            resources_per_trial={
                'gpu': 1.0,
            },
        )
    else:
        light_config = {
        }
        eval_config.update(light_config)
        _, config = next(tune.suggest.variant_generator.generate_variants(eval_config))
        with jax.disable_jit():
            trainable(config, reporter=lambda **x: x)
