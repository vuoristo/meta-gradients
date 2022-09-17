import os
import collections
import functools
import argparse
import itertools
from typing import Callable, Tuple, Dict, Union, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import rlax
from chex import Array, Scalar
import optax
from ray import tune
import pandas as pd
import numpy as np


Transition = collections.namedtuple(
    'Transition', [
        'observation',
        'logits',
        'action',
        'reward_tp1',
    ]
)

MetaFwdPassFnType = Callable[[Array, Array], Array]
LossFnType = Callable[[Array, Array, Transition], Tuple[Scalar, Scalar]]
OuterLossFnType = Callable[[Array, Array, Transition, Array, Array, Scalar,
                            Scalar, Any],
                           Tuple[Scalar, Scalar]]
SampleFnType = Callable[[Transition, Array], Transition]
SampleLifetimeFnType = Callable[..., Tuple[Scalar, Tuple[Transition, Transition]]]


def reset_env(batch_size: Scalar) -> Array:
    return jnp.zeros((batch_size, 1), dtype=jnp.int32)


def step_env(
    rngkey: Array,
    prev_obs: Array,
    action: Array,
    mus: Array,
    reward_noise_std: Scalar,
) -> Tuple[Array, Array]:
    rewards = mus[action] + jrandom.normal(rngkey, action.shape) * reward_noise_std
    obs = prev_obs + 1
    return obs, rewards


def sample_mus(
    rngkey: Array,
    num_actions: Scalar,
) -> Array:
    mus = jnp.exp(jrandom.uniform(rngkey, (num_actions,), minval=-100.0, maxval=1.0))
    return mus


def meta_forward_pass(
    eta: Array,
    state: Array,
    lifetime_length: Scalar,
    index_cutoff_ratio: Scalar,
) -> Array:
    def obs_to_index(x):
        i = jax.lax.select(x < lifetime_length // index_cutoff_ratio,
                           jnp.zeros_like(x, dtype=jnp.int32),
                           jnp.ones_like(x, dtype=jnp.int32))
        return jnp.array(i, dtype=int)
    index = obs_to_index(state.reshape(-1))
    inv_temps = jax.nn.softplus(eta)[index]
    leading_shape = state.shape[:-1]
    inv_temps = inv_temps.reshape(*leading_shape, 1)
    return inv_temps


def get_inv_temp_fn(
    config: Dict
) -> MetaFwdPassFnType:
    if config['inv_temp_fn_type'] == 'learned':
        return functools.partial(meta_forward_pass,
                                 lifetime_length=config['lifetime_length'],
                                 index_cutoff_ratio=config['index_cutoff_ratio'])
    elif config['inv_temp_fn_type'] == 'constant':
        def it_fn(eta: Array, state: Array) -> Array:
            return jnp.ones((*state.shape[:-1], 1, ))
        return it_fn
    else:
        raise ValueError


def get_learning_rate_fn(
    config: Dict
) -> MetaFwdPassFnType:
    if config['learning_rate_fn_type'] == 'learned':
        return functools.partial(meta_forward_pass,
                                 lifetime_length=config['lifetime_length'],
                                 index_cutoff_ratio=config['index_cutoff_ratio'])
    elif config['learning_rate_fn_type'] == 'constant':
        def lr_fn(eta: Array, state: Array) -> Array:
            return jnp.ones((*state.shape[:-1], 1, )
                            ) * config['inner_learning_rate']
        return lr_fn
    else:
        raise ValueError


def get_logits(
    lifetime_index: Array,
    theta: Array,
    eta: Array,
    inv_temp_fn: MetaFwdPassFnType,
) -> Array:
    leading_shape = lifetime_index.shape[:-1]
    logits_1 = jnp.broadcast_to(theta, leading_shape + theta.shape)
    inv_temps = inv_temp_fn(eta, lifetime_index)
    logits = logits_1 * inv_temps
    return logits


def sample_action(
    rngkey: Array,
    theta: Array,
    eta: Array,
    lifetime_index: Array,
    inv_temp_fn: MetaFwdPassFnType,
) -> Tuple[Array, Array]:
    logits = get_logits(lifetime_index, theta, eta, inv_temp_fn)
    action = jrandom.categorical(rngkey, logits)
    return action, logits


def surr_loss(
    theta: Array,
    eta: Array,
    lifetime_index: Array,
    transitions: Transition,
    inv_temp_fn: MetaFwdPassFnType,
) -> Tuple[Scalar, Scalar]:
    actions = transitions.action
    rewards = transitions.reward_tp1
    logits = get_logits(lifetime_index, theta, eta, inv_temp_fn)
    logpi = jax.nn.log_softmax(logits)
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(rewards * logpi_a)
    return pi_loss, jnp.ones_like(pi_loss)


def surr_loss_is(
    theta: Array,
    eta: Array,
    lifetime_index: Array,
    transitions: Transition,
    inv_temp_fn: MetaFwdPassFnType,
) -> Tuple[Scalar, Scalar]:
    actions = transitions.action
    rewards = transitions.reward_tp1
    behaviour_logits = transitions.logits
    logits = get_logits(lifetime_index, theta, eta, inv_temp_fn)
    rhos = rlax.categorical_importance_sampling_ratios(logits,
                                                       behaviour_logits,
                                                       actions)
    logpi = jax.nn.log_softmax(logits)
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(rewards * logpi_a)
    return pi_loss, rhos


def surr_loss_dice(
    theta: Array,
    eta: Array,
    lifetime_index: Array,
    transitions: Transition,
    inv_temp_fn: MetaFwdPassFnType,
) -> Tuple[Scalar, Scalar]:
    """ DiCE surrogate loss for bandits. Note that there is no
    time dimension."""
    actions = transitions.action
    rewards = transitions.reward_tp1
    logits = get_logits(lifetime_index, theta, eta, inv_temp_fn)
    logpi = jax.nn.log_softmax(logits)
    logpi_a = rlax.batched_index(logpi, actions)
    deps = jnp.exp(logpi_a - jax.lax.stop_gradient(logpi_a))
    pi_loss = -(rewards * deps).mean()
    return pi_loss, jnp.ones_like(pi_loss)


def batch_logpi_is(
    theta: Array,
    eta: Array,
    lifetime_index: Array,
    transitions: Transition,
    inv_temp_fn: MetaFwdPassFnType,
) -> Tuple[Array, Array]:
    actions = transitions.action
    behaviour_logits = transitions.logits
    logits = get_logits(lifetime_index, theta, eta, inv_temp_fn)
    rhos = rlax.categorical_importance_sampling_ratios(logits,
                                                       behaviour_logits,
                                                       actions)
    logpi = jax.nn.log_softmax(logits)
    logpi_a = rlax.batched_index(logpi, actions)
    return logpi_a, rhos


def batch_logpi(transitions: Transition) -> Array:
    actions = transitions.action
    logits = transitions.logits
    logpi = jax.nn.log_softmax(logits)
    logpi_a = rlax.batched_index(logpi, actions)
    return logpi_a


def get_surr_loss_fn(
    kind: Any,
    inv_temp_fn: MetaFwdPassFnType,
) -> LossFnType:
    if kind == 'is':
        return functools.partial(surr_loss_is, inv_temp_fn=inv_temp_fn)
    elif kind == 'no_is':
        return functools.partial(surr_loss, inv_temp_fn=inv_temp_fn)
    elif kind == 'dice':
        return functools.partial(surr_loss_dice, inv_temp_fn=inv_temp_fn)
    else:
        raise ValueError


def append_to_buffer(
    buffer: Union[Transition, Array],
    appendix: Union[Transition, Array],
    index: Array,
) -> Union[Transition, Array]:
    def appender(buf: Array, t: Array) -> Array:
        return buf.at[index].set(t)
    buffer = jax.tree_multimap(appender, buffer, appendix)
    return buffer


def sample_from_buffer(
    buffer: Transition,
    indices: Array,
) -> Transition:
    samples = jax.tree_map(lambda t: t[indices], buffer)
    return samples


def outer_loss_ignore_buffer(
    theta: Array,
    eta: Array,
    data_buffer: Transition,
    on_sample_index: Array,
    off_sample_index: Array,
    cum_logpi: Scalar,
    lifetime_index: Scalar,
    logpi_is_fn: Any,
    e_maml_lambda: Scalar,
) -> Tuple[Scalar, Scalar]:
    batch = sample_from_buffer(data_buffer, on_sample_index)
    local_logpi = e_maml_lambda * cum_logpi + batch_logpi(batch)
    loss = -jnp.mean(batch.reward_tp1 * local_logpi)
    logpi_out = batch_logpi(batch).sum()
    return loss, logpi_out


def outer_loss_off_policy(
    theta: Array,
    eta: Array,
    data_buffer: Transition,
    on_sample_index: Array,
    off_sample_index: Array,
    cum_logpi: Scalar,
    lifetime_index: Scalar,
    logpi_is_fn: Any,
    e_maml_lambda: Scalar,
) -> Tuple[Scalar, Scalar]:
    batch = sample_from_buffer(data_buffer, off_sample_index)
    lifetime_index = jnp.ones_like(batch.observation) * lifetime_index
    logpis, rhos = logpi_is_fn(theta, eta, lifetime_index, batch)
    rhos = jax.lax.stop_gradient(rhos)
    loss = -jnp.mean(batch.reward_tp1 * rhos * (logpis + e_maml_lambda * cum_logpi))
    logpi_out = (rhos * logpis).sum()
    return loss, logpi_out


def get_outer_loss_fn(kind: Any) -> OuterLossFnType:
    if kind == 'outer_loss_ignore_buffer':
        return outer_loss_ignore_buffer
    elif kind == 'outer_loss_off_policy':
        return outer_loss_off_policy
    else:
        raise ValueError


def get_random_indices(rngkey, buffer_size, lifetime_length,
                       append_batch_size, sample_batch_size):
    all_indices = jnp.arange(0, buffer_size)[None, :]
    modulos = (jnp.arange(1, lifetime_length + 1) * append_batch_size)[:, None]
    fill_indices = all_indices % modulos
    keys = jrandom.split(rngkey, lifetime_length)
    def sample_fn(rngkey, index):
        return jrandom.choice(rngkey, index, (sample_batch_size,))
    indices = jnp.array(list(map(sample_fn, keys, fill_indices)), dtype=jnp.int32)
    return indices


def get_logpi_cum_fn(config: Dict) -> Any:
    kind = config['logpi_cum_fn']
    if kind == 'meta_discount':
        meta_discount = config['meta_discount']
        def cum_logpi_fn(cum_logpi, outer_logpi):
            return (cum_logpi + outer_logpi) * meta_discount
        return cum_logpi_fn
    elif kind == 'meta_batch_size':
        batch_size = config['meta_batch_size']
        def cum_logpi_fn(cum_logpi, outer_logpi):
            return cum_logpi + outer_logpi / batch_size
        return cum_logpi_fn


def sample_lifetime(
    rngkey: Array,
    theta: Array,
    eta: Array,
    mus: Array,
    inv_temp_fn: MetaFwdPassFnType,
    lr_fn: MetaFwdPassFnType,
    loss_fn: LossFnType,
    sample_fn: SampleFnType,
    logpi_cum_fn: Any,
    config: Dict,
) -> Tuple[Scalar, Tuple[Transition, Transition]]:
    interact_batch_size = config['interact_batch_size']
    replay_batch_size = config['replay_batch_size']
    lt_len = config['lifetime_length']
    trunc_len = config['truncation_length']
    buffer_size = config['buffer_size']
    meta_discount = config['meta_discount']
    grad_fn = jax.vmap(jax.grad(loss_fn, has_aux=True), (None, None, 0, 0))
    logpi_is_fn = functools.partial(batch_logpi_is, inv_temp_fn=inv_temp_fn)
    outer_loss_fn = functools.partial(
        get_outer_loss_fn(config['outer_loss']),
        logpi_is_fn=logpi_is_fn,
        e_maml_lambda=config['e_maml_lambda'],
    )
    step_env_fn = functools.partial(
        step_env, reward_noise_std=config['reward_noise_std'])
    obs0 = reset_env(interact_batch_size)
    obs = obs0
    buffer = Transition(
        observation=jnp.zeros((buffer_size, 1), dtype=jnp.int32),
        logits=jnp.zeros((buffer_size, theta.shape[0])),
        action=jnp.zeros((buffer_size, ), dtype=jnp.int32),
        reward_tp1=jnp.zeros((buffer_size, )),
    )
    # Precreate indices for appending to replay buffer
    # and sampling from it. Required because jax.lax.scan
    # converts all inputs into abstract values, which breaks
    # aranges that depend on scan inputs inside the scan_fn
    append_indices = jnp.arange(0, buffer_size).reshape(
        lt_len, interact_batch_size)
    k1, k2 = jrandom.split(rngkey)
    if config['off_policy_inner'] == True:
        inner_sample_indices = get_random_indices(
            k2, buffer_size, lt_len, interact_batch_size,
            replay_batch_size)
    else:
        inner_sample_indices = append_indices
    if config['off_policy_outer'] == True:
        k1, k2 = jrandom.split(k1)
        outer_sample_indices = get_random_indices(
            k2, buffer_size, lt_len, interact_batch_size,
            replay_batch_size)
    else:
        outer_sample_indices = append_indices
    k1, k2 = jrandom.split(k1)
    trunc_start = jrandom.choice(k2, jnp.arange(0, lt_len - trunc_len + 1))
    pass_grad = jax.lax.dynamic_update_slice(
        jnp.zeros((lt_len,), dtype=jnp.bool_),
        jnp.ones((trunc_len,), dtype=jnp.bool_),
        (trunc_start,))
    k1, k2 = jrandom.split(k1)
    def scan_fn(prev_state, inputs):
        k1, theta, obs, buffer, cum_logpi = prev_state
        append_index, inner_sample_index, outer_sample_index, i = inputs
        local_theta = jax.lax.select(pass_grad[i], theta,
                                     jax.lax.stop_gradient(theta))
        local_eta = jax.lax.select(pass_grad[i], eta,
                                   jax.lax.stop_gradient(eta))
        cum_logpi = jax.lax.select(pass_grad[i], cum_logpi, 0.0)
        k1, k2 = jrandom.split(k1)
        lifetime_index = jnp.ones_like(obs) * i
        action, logits = sample_action(k2, local_theta, local_eta,
                                       lifetime_index, inv_temp_fn)
        k1, k2 = jrandom.split(k1)
        obs1, r = step_env_fn(k2, obs, action, mus)
        transition = Transition(
            observation=obs,
            logits=logits,
            action=action,
            reward_tp1=r,
        )
        obs = obs1
        buffer = append_to_buffer(buffer, transition, append_index)
        outer_loss, outer_logpi = outer_loss_fn(
            local_theta, local_eta, buffer, append_index, outer_sample_index,
            cum_logpi, i)
        cum_logpi = logpi_cum_fn(cum_logpi, outer_logpi)
        outer_loss = jax.lax.select(pass_grad[i], outer_loss, 0.0)
        sample = sample_fn(buffer, inner_sample_index)
        lifetime_index = jnp.ones_like(sample.observation) * i
        gs, is_ratios = grad_fn(local_theta, local_eta, lifetime_index, sample)
        g = jnp.mean(gs * is_ratios[:, None], axis=0)
        learning_rate = lr_fn(local_eta, lifetime_index[0, 0])
        local_theta = local_theta - learning_rate * g
        return (k1, local_theta, obs, buffer, cum_logpi), outer_loss
    carry, outputs = jax.lax.scan(
        scan_fn,
        (k1, theta, obs, buffer, 0.0),
        (append_indices, inner_sample_indices, outer_sample_indices,
         jnp.arange(lt_len))
    )
    k1, theta, obs, buffer, cum_logpi = carry
    cum_outer_loss = jnp.sum(outputs)
    cum_outer_loss = cum_outer_loss / float(trunc_len)
    # TODO: this is nasty
    truncated_buffer = jax.tree_map(
        lambda x: (x.reshape(lt_len, interact_batch_size, *x.shape[1:])
                   * pass_grad.reshape((-1, 1) + (1,) * (len(x.shape) - 1))
                   ).reshape(x.shape),
        buffer)
    return cum_outer_loss, (buffer, truncated_buffer)


def sample_lifetime_separate_buffers(
    rngkey: Array,
    theta: Array,
    eta: Array,
    mus: Array,
    inv_temp_fn: MetaFwdPassFnType,
    lr_fn: MetaFwdPassFnType,
    loss_fn: LossFnType,
    sample_fn: SampleFnType,
    config: Dict,
) -> Tuple[Scalar, Tuple[Transition, Transition]]:
    interact_batch_size = config['interact_batch_size']
    replay_batch_size = config['replay_batch_size']
    lt_len = config['lifetime_length']
    trunc_len = config['truncation_length']
    buffer_size = config['buffer_size']
    meta_discount = config['meta_discount']
    grad_fn = jax.vmap(jax.grad(loss_fn, has_aux=True), (None, None, 0, 0))
    logpi_is_fn = functools.partial(batch_logpi_is, inv_temp_fn=inv_temp_fn)
    outer_loss_fn = functools.partial(
        get_outer_loss_fn(config['outer_loss']),
        logpi_is_fn=logpi_is_fn,
    )
    obs0 = reset_env(interact_batch_size)
    obs = obs0
    buffers = Transition(
        observation=jnp.zeros((lt_len + 1, buffer_size, 1), dtype=jnp.int32),
        logits=jnp.zeros((lt_len + 1, buffer_size, theta.shape[0])),
        action=jnp.zeros((lt_len + 1, buffer_size, ), dtype=jnp.int32),
        reward_tp1=jnp.zeros((lt_len + 1, buffer_size, )),
    )
    append_indices = jnp.arange(0, buffer_size).reshape(
        lt_len, interact_batch_size)
    k1, k2 = jrandom.split(rngkey)
    inner_sample_indices = get_random_indices(
        k2, buffer_size, lt_len, interact_batch_size,
        replay_batch_size)
    k1, k2 = jrandom.split(k1)
    outer_sample_indices = get_random_indices(
        k2, buffer_size, lt_len, interact_batch_size,
        replay_batch_size)
    k1, k2 = jrandom.split(k1)
    trunc_start = jrandom.choice(k2, jnp.arange(0, lt_len - trunc_len + 1))
    pass_grad = jax.lax.dynamic_update_slice(
        jnp.zeros((lt_len,), dtype=jnp.bool_),
        jnp.ones((trunc_len,), dtype=jnp.bool_),
        (trunc_start,))
    cum_outer_loss = 0.0
    cum_logpi = 0.0
    def scan_fn(prev_state, inputs):
        k1, theta, obs, buffers, cum_outer_loss, cum_logpi = prev_state
        (append_index, inner_sample_index, outer_sample_index, i) = inputs
        lifetime_index = jnp.ones_like(obs) * i
        local_theta = jax.lax.select(pass_grad[i], theta,
                                     jax.lax.stop_gradient(theta))
        local_eta = jax.lax.select(pass_grad[i], eta,
                                   jax.lax.stop_gradient(eta))
        def sample_exp(rngkey, buffer):
            k1, k2 = jrandom.split(rngkey)
            action, logits = sample_action(k2, local_theta, local_eta,
                                           lifetime_index, inv_temp_fn)
            k1, k2 = jrandom.split(k1)
            obs1, r = step_env(k2, obs, action, mus)
            transition = Transition(
                observation=obs,
                logits=logits,
                action=action,
                reward_tp1=r,
            )
            buffer = append_to_buffer(buffer, transition, append_index)
            return buffer, obs1
        k1, k2 = jrandom.split(k1)
        keys = jrandom.split(k2, lt_len + 1)
        buffers, obs1 = jax.vmap(sample_exp)(keys, buffers)
        obs = obs1[0]
        outer_loss, outer_logpi = outer_loss_fn(
            local_theta, local_eta, jax.tree_map(lambda b: b[i], buffers),
            append_index, outer_sample_index, cum_logpi, i)
        cum_outer_loss += outer_loss
        cum_logpi += outer_logpi
        sample = sample_fn(jax.tree_map(lambda b: b[-1], buffers),
                           inner_sample_index)
        lifetime_index = jnp.ones_like(sample.observation) * i
        gs, is_ratios = grad_fn(local_theta, local_eta, lifetime_index, sample)
        g = jnp.mean(gs * is_ratios[:, None], axis=0)
        learning_rate = lr_fn(local_eta, lifetime_index[0, 0])
        local_theta = local_theta - learning_rate * g
        return (k1, local_theta, obs, buffers, cum_outer_loss, cum_logpi), None
    k1, k2 = jrandom.split(k1)
    carry, _ = jax.lax.scan(
        scan_fn,
        (k2, theta, obs, buffers, cum_outer_loss, cum_logpi),
        (append_indices, inner_sample_indices, outer_sample_indices,
         jnp.arange(lt_len))
    )
    k1, theta, obs, buffers, cum_outer_loss, cum_logpi = carry
    cum_outer_loss = cum_outer_loss / float(lt_len)
    out_buffer = jax.tree_map(lambda b: b[-1], buffers)
    truncated_buffer = jax.tree_map(
        lambda x: (x.reshape(lt_len, interact_batch_size, *x.shape[1:])
                   * pass_grad.reshape((-1, 1) + (1,) * (len(x.shape) - 1))
                   ).reshape(x.shape),
        out_buffer)
    return cum_outer_loss, (out_buffer, truncated_buffer)


def get_sample_lifetime_fn(config: Dict) -> SampleLifetimeFnType:
    if config['separate_buffers']:
        return sample_lifetime_separate_buffers
    else:
        return sample_lifetime


def compute_meta_gradient(
    rngkey: Array,
    theta: Array,
    eta: Array,
    mus: Array,
    inv_temp_fn: MetaFwdPassFnType,
    lr_fn: MetaFwdPassFnType,
    loss_fn: LossFnType,
    sample_fn: SampleFnType,
    logpi_cum_fn: Any,
    config: Dict,
) -> Array:
    sample_lifetime_fn = get_sample_lifetime_fn(config)
    inner_loop_fn = functools.partial(
        sample_lifetime_fn, inv_temp_fn=inv_temp_fn, loss_fn=loss_fn,
        sample_fn=sample_fn, lr_fn=lr_fn, logpi_cum_fn=logpi_cum_fn,
        config=config)
    def vectorized_inner_loop(*args):
        outer_loss, aux = jax.vmap(inner_loop_fn, (0, None, None, None))(*args)
        return outer_loss.mean(axis=0), aux
    keys = jrandom.split(rngkey, config['popsize'])
    grad_fn = jax.grad(vectorized_inner_loop, has_aux=True, argnums=2)
    g, aux = grad_fn(keys, theta, eta, mus)
    return g, aux


def compute_meta_gradient_fd(
    rngkey: Array,
    theta: Array,
    eta: Array,
    mus: Array,
    inv_temp_fn: MetaFwdPassFnType,
    lr_fn: MetaFwdPassFnType,
    loss_fn: LossFnType,
    sample_fn: SampleFnType,
    logpi_cum_fn: Any,
    config: Dict,
) -> Array:
    eps = config['epsilon']
    gs = []
    sample_lifetime_fn = get_sample_lifetime_fn(config)
    inner_loop_fn = functools.partial(
        sample_lifetime_fn, inv_temp_fn=inv_temp_fn, loss_fn=loss_fn,
        sample_fn=sample_fn, lr_fn=lr_fn, logpi_cum_fn=logpi_cum_fn,
        config=config)
    denom = config['truncation_length'] * config['interact_batch_size']
    for d in range(eta.shape[0]):
        unitvec = jax.nn.one_hot(d, eta.shape[0])
        perturbation = eps / 2. * unitvec
        plus_eta = eta + perturbation
        minus_eta = eta - perturbation
        _, plus_aux = inner_loop_fn(rngkey, theta, plus_eta, mus)
        plus_buffer = plus_aux[1]
        plus_returns = plus_buffer.reward_tp1.sum() / denom
        _, minus_aux = inner_loop_fn(rngkey, theta, minus_eta, mus)
        minus_buffer = minus_aux[1]
        minus_returns = minus_buffer.reward_tp1.sum() / denom
        gd = (plus_returns - minus_returns) / eps
        gs.append(gd)
    g = jnp.array(gs)
    _, out_buffers = inner_loop_fn(rngkey, theta, eta, mus)
    return g, out_buffers


def sample_gradients(
    rngkey: Array,
    eta: Array,
    config: Dict,
) -> Dict:
    K = config['K']
    parallel_runs = config['parallel_runs']
    compute_fd = config['compute_fd']
    compute_pg = config['compute_pg']
    update_with_fd = config['update_with_fd']
    update_with_pg = config['update_with_pg']
    inv_temp_fn = get_inv_temp_fn(config)
    lr_fn = get_learning_rate_fn(config)
    loss_fn = get_surr_loss_fn(config['inner_loss'], inv_temp_fn)
    sample_fn = sample_from_buffer
    logpi_cum_fn = get_logpi_cum_fn(config)
    compute_meta_gradient_ = jax.jit(jax.vmap(
        functools.partial(
            compute_meta_gradient,
            inv_temp_fn=inv_temp_fn,
            loss_fn=loss_fn,
            sample_fn=sample_fn,
            lr_fn=lr_fn,
            logpi_cum_fn=logpi_cum_fn,
            config=config,
        ), (0, None, None, 0)
    ))
    compute_meta_gradient_fd_ = jax.jit(jax.vmap(
        functools.partial(
            compute_meta_gradient_fd,
            inv_temp_fn=inv_temp_fn,
            loss_fn=loss_fn,
            sample_fn=sample_fn,
            lr_fn=lr_fn,
            logpi_cum_fn=logpi_cum_fn,
            config=config,
        ), (0, None, None, 0)
    ))
    sample_mus_ = jax.jit(jax.vmap(functools.partial(sample_mus, num_actions=K)))
    logdict = collections.defaultdict(list)
    keys = jrandom.split(rngkey, 2)
    if config['opt_type'] == 'adam':
        opt_kwargs = config['opt_kwargs'].copy()
        learning_rate = opt_kwargs.pop('learning_rate')
        optimizer = optax.chain(
            optax.scale_by_adam(**opt_kwargs),
            optax.scale(-learning_rate),
        )
    elif config['opt_type'] == 'sgd':
        opt_kwargs = config['opt_kwargs'].copy()
        learning_rate = opt_kwargs.pop('learning_rate')
        optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(eta)
    for iteration in range(config['iterations']):
        keys = jrandom.split(keys[0], parallel_runs + 1)
        mus = sample_mus_(keys[1:])
        theta = jnp.zeros((K), dtype=jnp.float32)
        if compute_fd:
            fd_g, aux = compute_meta_gradient_fd_(keys[1:], theta, eta, mus)
            buffer = jax.device_get(jax.tree_map(jnp.mean, aux))[0]
            mean_fd_g = jax.device_get(fd_g.mean(axis=0))
            logdict['eta_fd_g0'].append(mean_fd_g[0])
            logdict['eta_fd_g1'].append(mean_fd_g[1])
            var_fd_g = jax.device_get(fd_g.var(axis=0))
            logdict['eta_fd_g0_var'].append(var_fd_g[0])
            logdict['eta_fd_g1_var'].append(var_fd_g[1])
        if compute_pg:
            pg_g, aux = compute_meta_gradient_(keys[1:], theta, eta, mus)
            buffer = jax.device_get(jax.tree_map(jnp.mean, aux))[0]
            mean_pg_g = -jax.device_get(pg_g.mean(axis=0))
            logdict['eta_pg_g0'].append(mean_pg_g[0])
            logdict['eta_pg_g1'].append(mean_pg_g[1])
            var_pg_g = jax.device_get(pg_g.var(axis=0))
            logdict['eta_pg_g0_var'].append(var_pg_g[0])
            logdict['eta_pg_g1_var'].append(var_pg_g[1])
        logdict['iteration'].append(iteration)
        logdict['returns'].append(buffer.reward_tp1)
        if update_with_fd:
            assert not update_with_pg
            updates, opt_state = optimizer.update(-1.0 * fd_g.mean(axis=0), opt_state)
            eta = optax.apply_updates(eta, updates)
        if update_with_pg:
            assert not update_with_fd
            updates, opt_state = optimizer.update(pg_g.mean(axis=0), opt_state)
            eta = optax.apply_updates(eta, updates)
        if update_with_fd or update_with_pg:
            for i, x in enumerate(jax.device_get(eta)):
                logdict[f'eta{i}'].append(x)
                logdict[f'temp{i}'].append(1/np.logaddexp(x, 0))
    return logdict


def trainable(
    config: Dict,
    checkpoint_dir: Any=None,
):
    rngkey = jrandom.PRNGKey(config['seed'])
    eta = jnp.array(config['initial_eta'])
    logdict = sample_gradients(rngkey, eta, config)
    with tune.checkpoint_dir(step=1) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "results.csv")
        pd.DataFrame(logdict).to_csv(path)
    tune.report(returns=np.mean(logdict['returns']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ray')
    args = parser.parse_args()

    side = jax.device_get(np.linspace(0.0, 2.0, 10))
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
        reward_noise_std=1.0,
        inner_learning_rate=tune.grid_search([20.0]),
        lifetime_length=tune.grid_search([10]),
        index_cutoff_ratio=3,
        # truncation_length=5,
        truncation_length=tune.grid_search([2, 5, 10]),
        opt_type='sgd',
        opt_kwargs=dict(learning_rate=1e-1),
        meta_discount=tune.grid_search([1.0, 0.0]),
        # meta_discount=1.0,
        e_maml_lambda=0.1,
        logpi_cum_fn=tune.grid_search([
            'meta_discount',
            'emaml',
        ]),
        inner_loss=tune.grid_search([
            'is',
            'no_is',
        ]),
        outer_loss=tune.grid_search([
            'outer_loss_ignore_buffer',
            'outer_loss_off_policy',
        ]),
        separate_buffers=tune.grid_search([
            True,
            False,
        ]),
        off_policy_inner=False,
        off_policy_outer=False,
        seed=0,
        epsilon=1e-1,
        iterations=10000,
        parallel_runs=100,
        fd=tune.grid_search([True, False]),
        compute_fd=tune.sample_from(lambda spec: spec.get('config', spec)['fd']),
        compute_pg=tune.sample_from(lambda spec: not spec.get('config', spec)['fd']),
        update_with_fd=tune.sample_from(lambda spec: spec.get('config', spec)['fd']),
        update_with_pg=tune.sample_from(lambda spec: not spec.get('config', spec)['fd']),
    )

    if args.mode == 'ray':
        analysis = tune.run(
            trainable,
            name='divide_by_lifetime_len',
            config=config,
            verbose=1,
            resources_per_trial={
                'gpu': 0.3,
            },
        )
    elif args.mode == 'ray_single':
        _, config = next(tune.suggest.variant_generator.generate_variants(config))
        analysis = tune.run(
            trainable,
            name='single',
            config=config,
            resources_per_trial={
                'gpu': 1.0,
            },
        )
    else:
        light_config = {}
        config.update(light_config)
        _, config = next(tune.suggest.variant_generator.generate_variants(config))
        with jax.disable_jit():
            trainable(config)
