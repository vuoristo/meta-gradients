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


Array = chex.Array
Scalar = chex.Scalar
Numeric = chex.Numeric


TimeStep = collections.namedtuple(
    'TimeStep', [
        'action_tm1',
        'reward',
        'discount',
        'observation',
        'episode_length',
        'lifetime_length',
        'goals',
    ]
)

Log = collections.namedtuple(
    'Log', [
        'pi_loss',
        'baseline_loss',
        'entropy',
        'grad_norm',
        'ent_coef',
        'advantage',
        'learning_rate',
        'eta_output',
    ]
)

MetaFwdPassFnType = Callable[[Array, Array], Array]
LossFnType = Callable[[Array, TimeStep, Scalar], Tuple[Scalar, Array]]
OuterLossFnType = Callable[[Array, Array, TimeStep, Array, Array, Scalar,
                            Scalar, Any],
                           Tuple[Scalar, Scalar]]
SampleFnType = Callable[[TimeStep, Array], TimeStep]
SampleLifetimeFnType = Callable[..., Tuple[Scalar, Tuple[TimeStep, TimeStep]]]


class ActorCriticNet(hk.Module):
    def __init__(self, num_actions, name=None, **kwargs):
        super(ActorCriticNet, self).__init__(name=name)
        self._num_actions = num_actions

    def __call__(self, inputs):
        def torso(x):
            net = hk.Sequential([
                hk.Flatten(), hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu])
            return net(x)
        pi_h = torso(inputs)
        logits = hk.Linear(self._num_actions)(pi_h)
        v_h = torso(inputs)
        v = hk.Linear(1)(v_h).squeeze(-1)
        return logits, v


class ActorNet(hk.Module):
    def __init__(self, num_actions, name=None, **kwargs):
        super(ActorNet, self).__init__(name=name)
        self._num_actions = num_actions

    def __call__(self, inputs):
        def torso(x):
            net = hk.Sequential([
                hk.Flatten(), hk.Linear(256), jax.nn.relu,
                hk.Linear(256), jax.nn.relu])
            return net(x)
        pi_h = torso(inputs)
        logits = hk.Linear(self._num_actions)(pi_h)
        v = jnp.zeros(logits.shape[:-1])
        return logits, v


class EtaNetSimple(hk.Module):
    def __init__(self, initial_eta, name=None, **kwargs):
        super(EtaNetSimple, self).__init__(name=name)
        self._initial_eta = initial_eta

    def get_initial_state(self, batch_size):
        return None

    def __call__(self, inputs, _):
        eta = hk.get_parameter('w', [], init=hk.initializers.Constant(self._initial_eta))
        output = jax.nn.sigmoid(jnp.ones(inputs.shape[:-1]) * eta)
        return output


class EtaNetContext(hk.Module):
    def __init__(self, num_outputs=1, name=None, **kwargs):
        super(EtaNetContext, self).__init__(name=name)
        self._num_outputs = num_outputs

    def get_initial_state(self, batch_size):
        return None

    def __call__(self, inputs, _):
        net = hk.Sequential([
            hk.Flatten(),
            hk.Linear(32),
            jax.nn.relu,
            hk.Linear(self._num_outputs),
            jax.nn.sigmoid,
        ])
        output = net(inputs)
        return output


class EtaNetFrodo(hk.Module):
    def __init__(self, input_keys, name=None):
        super(EtaNetFrodo, self).__init__(name=name)
        self._core = hk.ResetCore(hk.LSTM(256))
        self._input_keys = input_keys

    def get_initial_state(self, batch_size):
        return self._core.initial_state(batch_size)

    def __call__(self, inputs, state):
        should_reset = jnp.array(1 - inputs['discount'], dtype=bool)
        # T, B, -1
        inputs = jax.tree_map(
            lambda x: x.reshape(inputs['discount'].shape + (-1,)), inputs)
        lstm_inputs = jnp.concatenate(
            [inputs[k] for k in self._input_keys], axis=-1)
        rnn_inputs = (lstm_inputs, should_reset)
        rnn_inputs = jax.tree_map(lambda x: x[::-1], rnn_inputs)
        # [T, H], unroll over time.
        core_output, state = hk.dynamic_unroll(self._core, rnn_inputs, state)
        output = hk.BatchApply(hk.Linear(
                1, w_init=hk.initializers.TruncatedNormal(stddev=1e-3),
            ))(core_output)[::-1].squeeze(-1)
        return output, state


def get_eta_net(config: Dict) -> hk.Module:
    if config['eta_net'] == 'simple':
        return EtaNetSimple
    elif config['eta_net'] == 'context':
        return EtaNetContext
    elif config['eta_net'] == 'frodo':
        return EtaNetFrodo
    else:
        raise ValueError()


def step_env(
    rngkey: Array,
    timestep: TimeStep,
    a: Array,
    batch_size: Scalar,
    step_env_fn: Any,
    reset_env_fn: Any,
    flip_steps: int,
) -> TimeStep:
    keys = jrandom.split(rngkey, batch_size + 1)
    s = timestep.observation
    goals = timestep.goals
    s, r, terminate = jax.vmap(step_env_fn, (0, 0, 0, 0))(keys[:-1], s, a, goals)
    ep_len = (timestep.episode_length + 1) * (1 - terminate)
    lt_len = timestep.lifetime_length + 1
    # Reset finished envs
    keys = jrandom.split(keys[-1], batch_size)
    new_s = jax.vmap(reset_env_fn)(keys)
    flag = jnp.array(
        jnp.expand_dims(terminate, axis=tuple((range(1, s.ndim))))
        * jnp.ones_like(s, dtype=bool),
        dtype=bool)
    out_s = jax.lax.select(flag, new_s, s)
    ts = TimeStep(
        action_tm1=a,
        reward=r,
        discount=1 - terminate,
        observation=out_s,
        episode_length=ep_len,
        lifetime_length=lt_len,
        goals=goals,
    )
    return ts


def sample_action(
    rngkey: Array,
    theta: Array,
    state: Array,
    apply_theta_fn: Any,
) -> Array:
    logits, _ = apply_theta_fn(theta, state)
    a = hk.multinomial(rngkey, logits, num_samples=1).squeeze(axis=-1)
    return a


def surr_loss(
    theta: Array,
    trajs: TimeStep,
    ent_coef: Scalar,
    gamma: Scalar,
    vf_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(advantage * logpi_a)
    baseline_loss = 0.5 * jnp.mean(jnp.square(returns - v_tm1))
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + vf_coef * baseline_loss + ent_coef * ent_loss
    log = Log(pi_loss, baseline_loss, -ent_loss, 0.0, ent_coef, advantage, 0.0, 0.0)
    return total_loss, log


def frodo_loss(
    theta: Array,
    eta: Any,
    trajs: TimeStep,
    ent_coef: Scalar,
    gamma: Scalar,
    vf_coef: Scalar,
    apply_theta_fn: Any,
    apply_eta_fn: Any,
    init_eta_state_fn: Any,
) -> Tuple[Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    eta_inputs = jax.lax.stop_gradient(dict(
        reward=rewards,
        discount=discounts,
        pi_a=jnp.exp(logpi_a),
        v_t=v_t,
    ))
    eta_state = init_eta_state_fn(None, actions.shape[1])
    returns, _ = apply_eta_fn(eta, eta_inputs, eta_state)
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    pi_loss = -jnp.mean(advantage * logpi_a)
    baseline_loss = 0.5 * jnp.mean(jnp.square(returns - v_tm1))
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + vf_coef * baseline_loss + ent_coef * ent_loss
    log = Log(pi_loss, baseline_loss, -ent_loss, 0.0, ent_coef, advantage, 0.0, returns)
    return total_loss, log


def surr_loss_no_baseline(
    theta: Array,
    trajs: TimeStep,
    ent_coef: Scalar,
    gamma: Scalar,
    vf_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(returns * logpi_a)
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    baseline_loss = 0.0
    total_loss = pi_loss + vf_coef * baseline_loss + ent_coef * ent_loss
    log = Log(pi_loss, baseline_loss, -ent_loss, 0.0, ent_coef, returns, 0.0, 0.0)
    return total_loss, log


def vf_loss(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    _, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    baseline_loss = 0.5 * jnp.mean(jnp.square(returns - v_tm1))
    return baseline_loss


def outer_loss_vanilla(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    local_logpi = sc_coef * cum_logpi + logpi_a
    pi_loss = -jnp.mean(advantage * local_logpi)
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, advantage, 0.0, 0.0)
    return total_loss, logpi_a.sum(), log


def outer_loss_avg_return(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = - logpi_a.sum() * jnp.mean(returns)
    sampling_correction = -cum_logpi * jnp.mean(returns)
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss + sc_coef * sampling_correction
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, returns, 0.0)
    return total_loss, logpi_a.sum(), log


def outer_loss_first_state_return(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = - jnp.sum(logpi_a * returns, axis=1).mean()
    sampling_correction = -cum_logpi * jnp.mean(returns[:, :1])
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss + sc_coef * sampling_correction
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, returns[:, :1], 0.0, 0.0)
    return total_loss, logpi_a.sum(), log


def outer_loss_first_state_advantage(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = - jnp.sum(logpi_a * advantage, axis=1).mean()
    sampling_correction = -cum_logpi * jnp.mean(advantage[:, :1])
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss + sc_coef * sampling_correction
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, returns[:, :1], 0.0, 0.0)
    return total_loss, logpi_a.sum(), log


def outer_loss_standard_no_baseline(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = - jnp.mean(logpi_a * returns)
    sampling_correction = -cum_logpi * jnp.mean(returns)
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss + sc_coef * sampling_correction
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, returns, 0.0)
    return total_loss, logpi_a.sum(), log


def outer_loss_direct_baseline(
    theta: Array,
    trajs: TimeStep,
    gamma: Scalar,
    cum_logpi: Scalar,
    ent_coef: Scalar,
    sc_coef: Scalar,
    apply_theta_fn: Any,
) -> Tuple[Scalar, Scalar, Log]:
    rewards = trajs.reward[:, 1:]
    discounts = trajs.discount[:, 1:] * gamma
    observations = trajs.observation
    logits, value = jax.vmap(apply_theta_fn, (None, 0))(theta, observations)
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    actions = trajs.action_tm1[:, 1:]
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(advantage * logpi_a)
    sampling_correction = -cum_logpi * jnp.mean(returns)
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + ent_coef * ent_loss + sc_coef * sampling_correction
    log = Log(pi_loss, 0.0, -ent_loss, 0.0, 0.0, returns, 0.0)
    return total_loss, logpi_a.sum(), log


def get_outer_loss_fn(
    config: Dict,
) -> Callable:
    kind = config['outer_loss']
    if kind == 'vanilla':
        return outer_loss_vanilla
    elif kind == 'avg_return':
        return outer_loss_avg_return
    elif kind == 'first_state':
        return outer_loss_first_state_return
    elif kind == 'standard_no_baseline':
        return outer_loss_standard_no_baseline
    elif kind == 'direct_baseline':
        return outer_loss_direct_baseline
    elif kind == 'first_state_adv':
        return outer_loss_first_state_advantage
    else:
        raise ValueError


def sample_trajs(
    rngkey: Array,
    theta: Array,
    timestep: TimeStep,
    step_env_fn: Any,
    reset_env_fn: Any,
    apply_theta_fn: Any,
    num_trajs: Scalar,
    H: Scalar,
    flip_steps: int,
) -> TimeStep:
    step = functools.partial(step_env, batch_size=num_trajs,
                             step_env_fn=step_env_fn,
                             reset_env_fn=reset_env_fn,
                             flip_steps=flip_steps)
    sample_action_ = functools.partial(sample_action, apply_theta_fn=apply_theta_fn)
    def scan_f(prev_state, _):
        k1, timestep = prev_state
        obs = timestep.observation
        k1, k2 = jrandom.split(k1)
        a = sample_action_(k2, theta, obs)
        k1, k2 = jrandom.split(k1)
        ts = step(k2, timestep, a)
        return (k1, ts), ts
    _, traj_post = jax.lax.scan(scan_f, (rngkey, timestep), jnp.arange(H))
    traj_pre = pack_namedtuple_jnp([timestep])
    traj = jax.tree_multimap(lambda *xs: jnp.moveaxis(jnp.concatenate(xs), 0, 1),
                             traj_pre, traj_post)
    return traj


def get_scan_fn(
    eta: Array,
    inner_optimizer: Any,
    sample_fn: Any,
    apply_theta_fn: Any,
    apply_eta_fn: Any,
    init_eta_state_fn: Any,
    outer_loss_fn: Any,
    config: Dict,
) -> Callable:
    gamma = config['gamma']
    if config['inner_loss'] == 'no_is':
        grad_fn = jax.grad(functools.partial(
            surr_loss, apply_theta_fn=apply_theta_fn), has_aux=True)
    elif config['inner_loss'] == 'no_is_no_baseline':
        grad_fn = jax.grad(functools.partial(
            surr_loss_no_baseline, apply_theta_fn=apply_theta_fn), has_aux=True)
    elif config['inner_loss'] == 'frodo':
        grad_fn = jax.grad(functools.partial(
            frodo_loss, apply_theta_fn=apply_theta_fn,
            apply_eta_fn=apply_eta_fn, init_eta_state_fn=init_eta_state_fn),
            has_aux=True)
    if config['inner_loop_fn'] == 'default':
        vf_coef = config['vf_coef']
        def scan_fn(prev_state, _):
            k1, theta, theta_opt_state, timestep, cum_logpi, eta_inputs = prev_state
            k1, k2 = jrandom.split(k1)
            trajs = sample_fn(k2, theta, timestep)
            outer_loss, outer_logpi, outer_log = outer_loss_fn(theta, trajs, gamma, cum_logpi)
            cum_logpi += outer_logpi
            new_eta_inputs = jnp.concatenate([eta_inputs[1:], trajs.reward.mean()[None]])
            ent_coef = apply_eta_fn(eta, new_eta_inputs, None).squeeze(-1)
            grad, inner_log = grad_fn(theta, trajs, ent_coef=ent_coef,
                                      vf_coef=vf_coef, gamma=gamma)
            inner_log = inner_log._replace(grad_norm=optax.global_norm(grad))
            updates, theta_opt_state = inner_optimizer.update(grad, theta_opt_state, params=theta)
            theta = optax.apply_updates(theta, updates)
            timestep = jax.tree_map(lambda x: x[:, -1], trajs)
            rewards_out = trajs.reward[:, 1:]
            new_state = (k1, theta, theta_opt_state, timestep, cum_logpi, new_eta_inputs)
            outputs = (outer_loss, rewards_out, outer_log, inner_log)
            return new_state, outputs
    elif config['inner_loop_fn'] == 'frodo':
        vf_coef = config['vf_coef']
        ent_coef = config['ent_coef']
        def scan_fn(prev_state, _):
            k1, theta, theta_opt_state, timestep, cum_logpi, eta_inputs = prev_state
            k1, k2 = jrandom.split(k1)
            trajs = sample_fn(k2, theta, timestep)
            outer_loss, outer_logpi, outer_log = outer_loss_fn(theta, trajs, gamma,
                                                               cum_logpi)
            cum_logpi += outer_logpi
            grad, inner_log = grad_fn(theta, eta, trajs, vf_coef=vf_coef,
                                      gamma=gamma, ent_coef=ent_coef)
            inner_log = inner_log._replace(grad_norm=optax.global_norm(grad))
            updates, theta_opt_state = inner_optimizer.update(grad, theta_opt_state,
                                                              params=theta)
            theta = optax.apply_updates(theta, updates)
            timestep = jax.tree_map(lambda x: x[:, -1], trajs)
            rewards_out = trajs.reward[:, 1:]
            mean_rewards = jnp.concatenate([eta_inputs[1:], trajs.reward.mean()[None]])
            new_state = (k1, theta, theta_opt_state, timestep, cum_logpi, mean_rewards)
            outputs = (outer_loss, rewards_out, outer_log, inner_log)
            return new_state, outputs
    elif config['inner_loop_fn'] == 'separate_value':
        vf_coef = config['vf_coef']
        policy_loss = functools.partial(surr_loss, vf_coef=0.0)
        vf_grad_fn = jax.grad(functools.partial(
            vf_loss, apply_theta_fn=apply_theta_fn))
        def scan_fn(prev_state, _):
            k1, theta, theta_opt_state, timestep, cum_logpi, eta_inputs = prev_state
            k1, k2 = jrandom.split(k1)
            trajs = sample_fn(k2, theta, timestep)
            outer_loss, outer_logpi, outer_log = outer_loss_fn(theta, trajs, gamma, cum_logpi)
            cum_logpi += outer_logpi
            new_eta_inputs = jnp.concatenate([eta_inputs[1:], trajs.reward.mean()[None]])
            ent_coef = apply_eta_fn(eta, new_eta_inputs, None).squeeze(-1)
            grad, inner_log = grad_fn(theta, trajs, ent_coef=ent_coef,
                                      vf_coef=0.0, gamma=gamma)
            k1, k2 = jrandom.split(k1)
            vf_trajs = sample_fn(k2, theta, timestep)
            vf_grad = vf_grad_fn(theta, vf_trajs, gamma=gamma)
            grad = jax.tree_multimap(lambda g1, g2: (g1 + vf_coef * g2), grad, vf_grad)
            inner_log = inner_log._replace(grad_norm=optax.global_norm(grad))
            updates, theta_opt_state = inner_optimizer.update(grad, theta_opt_state, params=theta)
            theta = optax.apply_updates(theta, updates)
            timestep = jax.tree_map(lambda x: x[:, -1], trajs)
            rewards_out = trajs.reward[:, 1:]
            new_state = (k1, theta, theta_opt_state, timestep, cum_logpi, new_eta_inputs)
            outputs = (outer_loss, rewards_out, outer_log, inner_log)
            return new_state, outputs
    elif config['inner_loop_fn'] == 'learn_lr_ent_coef':
        vf_coef = config['vf_coef']
        policy_loss = functools.partial(surr_loss, vf_coef=0.0)
        vf_grad_fn = jax.grad(functools.partial(
            vf_loss, apply_theta_fn=apply_theta_fn))
        def scan_fn(prev_state, _):
            k1, theta, theta_opt_state, timestep, cum_logpi, eta_inputs = prev_state
            k1, k2 = jrandom.split(k1)
            trajs = sample_fn(k2, theta, timestep)
            outer_loss, outer_logpi, outer_log = outer_loss_fn(theta, trajs, gamma, cum_logpi)
            cum_logpi += outer_logpi
            new_eta_inputs = jnp.concatenate([eta_inputs[1:], trajs.reward.mean()[None]])
            ent_coef, lr = apply_eta_fn(eta, new_eta_inputs, None)
            grad, inner_log = grad_fn(theta, trajs, ent_coef=ent_coef,
                                      vf_coef=0.0, gamma=gamma)
            inner_log = inner_log._replace(grad_norm=optax.global_norm(grad), learning_rate=lr)
            k1, k2 = jrandom.split(k1)
            vf_trajs = sample_fn(k2, theta, timestep)
            vf_grad = vf_grad_fn(theta, vf_trajs, gamma=gamma)
            grad = jax.tree_multimap(lambda g1, g2: (g1 + vf_coef * g2), grad, vf_grad)
            theta_opt_state[1].hyperparams['learning_rate'] = lr
            updates, theta_opt_state = inner_optimizer.update(grad, theta_opt_state, params=theta)
            theta = optax.apply_updates(theta, updates)
            timestep = jax.tree_map(lambda x: x[:, -1], trajs)
            rewards_out = trajs.reward[:, 1:]
            new_state = (k1, theta, theta_opt_state, timestep, cum_logpi, new_eta_inputs)
            outputs = (outer_loss, rewards_out, outer_log, inner_log)
            return new_state, outputs
    else:
        raise ValueError()
    return scan_fn


def inner_unroll(
    rngkey: Array,
    theta: Array,
    eta: Array,
    mean_rewards: Array,
    timestep: TimeStep,
    step_env_fn: Any,
    reset_env_fn: Any,
    apply_theta_fn: Any,
    apply_eta_fn: Any,
    init_theta_fn: Any,
    resample_goals_fn: Any,
    inner_optimizer: Any,
    init_eta_state_fn: Any,
    config: Dict,
) -> Tuple[Scalar, Tuple[TimeStep, TimeStep]]:
    num_trajs = config['inner_batch_size']
    flip_interval = config['reward_flip_interval']
    trunc_len = config['truncation_length']
    auc_loss = config['auc_loss']
    reset_theta_at_reward_flip = config['reset_theta_at_reward_flip']
    sample_fn = functools.partial(
        sample_trajs,
        step_env_fn=step_env_fn,
        reset_env_fn=reset_env_fn,
        apply_theta_fn=apply_theta_fn,
        num_trajs=num_trajs,
        H=config['H'],
        flip_steps=config['reward_flip_interval'],
    )
    outer_loss_fn = functools.partial(
        get_outer_loss_fn(config),
        ent_coef=config['ent_coef'],
        sc_coef=config['meta_lambda'],
        apply_theta_fn=apply_theta_fn,
    )
    scan_fn = get_scan_fn(eta, inner_optimizer, sample_fn, apply_theta_fn,
                          apply_eta_fn, init_eta_state_fn, outer_loss_fn, config)
    k1, k2 = jrandom.split(rngkey)
    lt_len = timestep.lifetime_length[0]
    should_flip = lt_len >= flip_interval
    if reset_theta_at_reward_flip:
        new_theta = init_theta_fn(k2, timestep.observation)
        theta = jax.tree_multimap(
            lambda a, b: jax.lax.select(should_flip, a, b), new_theta, theta)
        mean_rewards = jax.tree_multimap(
            lambda a, b: jax.lax.select(should_flip, a, b),
            jnp.zeros_like(mean_rewards), mean_rewards)
    # Randomize goals if flip interval reached
    k1, k2 = jrandom.split(k1)
    new_goals = resample_goals_fn(k2)
    new_goals = new_goals[None].tile((num_trajs, *(1, ) * (new_goals.ndim)))
    goals = jax.lax.select(should_flip, new_goals, timestep.goals)
    lt_len = jax.lax.select(
        should_flip,
        jnp.zeros_like(timestep.lifetime_length),
        timestep.lifetime_length
    )
    timestep = timestep._replace(
        lifetime_length=lt_len,
        goals=goals,
    )
    # TODO: this should carry between inner_unrolls but
    # the optax optimizer states do not vmap nicely
    theta_opt_state = inner_optimizer.init(theta)
    carry, outputs = jax.lax.scan(
        scan_fn,
        (k1, theta, theta_opt_state, timestep, 0.0, mean_rewards),
        jnp.arange(trunc_len)
    )
    k1, theta, theta_opt_state, timestep, cum_logpi, mean_rewards = carry
    outer_loss, rewards, outer_log, inner_log = outputs
    mask_len = trunc_len - 1
    outer_loss_mask = jax.lax.select(
        auc_loss * jnp.ones(mask_len, dtype=bool),
        jnp.ones(mask_len), jnp.zeros(mask_len))
    outer_loss_mask = jnp.concatenate((outer_loss_mask, jnp.ones(1)))
    cum_outer_loss = jnp.sum(outer_loss * outer_loss_mask) / outer_loss_mask.sum()
    return cum_outer_loss, (rewards, theta, mean_rewards, timestep,
                            outer_log, inner_log)


def train_step_mg(
    key: Array,
    eta: Array,
    theta: Array,
    mean_rewards: Array,
    timestep: TimeStep,
    eta_opt_state: Any,
    inner_loop_fn: Any,
    eta_optimizer: Any,
) -> Array:
    def vectorized_inner_loop(*args):
        outer_loss, outputs = jax.vmap(inner_loop_fn, (0, 0, None, 0, 0))(*args)
        return outer_loss.mean(), outputs
    grad_fn = jax.grad(vectorized_inner_loop, has_aux=True, argnums=2)
    keys = jrandom.split(key, mean_rewards.shape[0])
    g, (rewards_out, new_theta, new_mean_rewards, new_timestep, outer_log,
        inner_log) = grad_fn(keys, theta, eta, mean_rewards, timestep)
    outer_log = outer_log._replace(grad_norm=optax.global_norm(g))
    updates, new_eta_opt_state = eta_optimizer.update(g, eta_opt_state, params=eta)
    new_eta = optax.apply_updates(eta, updates)
    # check for nans and skip this update if there are any
    theta_nan = jnp.any(jnp.stack(
        jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(new_theta))))
    eta_nan = jnp.any(jnp.stack(
        jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), jax.tree_leaves(new_eta))))
    has_nan = theta_nan | eta_nan
    nan_replace = lambda a, b: jnp.where(has_nan, a, b)
    new_mean_rewards = nan_replace(mean_rewards, new_mean_rewards)
    new_theta = jax.tree_multimap(nan_replace, theta, new_theta)
    new_eta = jax.tree_multimap(nan_replace, eta, new_eta)
    new_eta_opt_state = jax.tree_multimap(nan_replace, eta_opt_state, new_eta_opt_state)
    return (new_eta, new_theta, rewards_out, new_mean_rewards, new_timestep, new_eta_opt_state,
            outer_log, inner_log, g, theta_nan, eta_nan, theta, eta, eta_opt_state)


def train_step_fd(
    key: Array,
    eta: Array,
    theta: Array,
    mean_rewards: Array,
    timestep: TimeStep,
    eta_opt_state: Any,
    inner_loop_fn: Any,
    eta_optimizer: Any,
    fd_epsilon: Scalar,
) -> Array:
    def vectorized_inner_loop(*args):
        _, outputs = jax.vmap(inner_loop_fn, (0, 0, None, 0, 0))(*args)
        return outputs
    gs = []
    params, treedef = jax.tree_flatten(eta)
    k1 = key
    for i, p in enumerate(params):
        unitvec = jnp.ones_like(p)
        perturbation = fd_epsilon / 2. * unitvec
        p_p = p + perturbation
        p_eta = jax.tree_unflatten(treedef, params[:i] + [p_p] + params[i+1:])
        k1, k2 = jrandom.split(k1)
        keys = jrandom.split(k2, mean_rewards.shape[0])
        (_, _, _, _, p_outer_log, _) = vectorized_inner_loop(
            keys, theta, p_eta, mean_rewards, timestep)
        m_p = p - perturbation
        m_eta = jax.tree_unflatten(treedef, params[:i] + [m_p] + params[i+1:])
        (_, _, _, _, m_outer_log,_) = vectorized_inner_loop(
            keys, theta, m_eta, mean_rewards, timestep)
        gd = -1.0 * (p_outer_log.advantage.mean()
                     - m_outer_log.advantage.mean()) / fd_epsilon
        gs.append(gd)
    (rewards, theta, mean_rewards, timestep, outer_log,
        inner_log) = vectorized_inner_loop(keys, theta, eta, mean_rewards, timestep)
    g = jax.tree_unflatten(treedef, gs)
    outer_log = outer_log._replace(grad_norm=optax.global_norm(g))
    updates, eta_opt_state = eta_optimizer.update(g, eta_opt_state)
    eta = optax.apply_updates(eta, updates)
    return (eta, theta, rewards, mean_rewards, timestep, eta_opt_state,
            outer_log, inner_log, g)


def get_train_step_fn(
    config: Dict
) -> Callable:
    if config['train_step'] == 'mg':
        return train_step_mg
    elif config['train_step'] == 'fd':
        return functools.partial(train_step_fd, fd_epsilon=config['fd_epsilon'])
    else:
        raise ValueError()


def log_status(
    reporter: Any,
    logdict: collections.defaultdict,
    config: Dict,
    log_steps: set,
    eval_steps: set,
    iteration: int,
    total_reward: float,
    total_steps: int,
    rewards: Array,
    inner_log: Log,
    outer_log: Log,
    eta_opt_state: Any,
    theta_nan: bool,
    eta_nan: bool,
    theta: Any,
    eta: Any,
) -> collections.defaultdict:
    outer_log = jax.device_get(outer_log)
    inner_log = jax.device_get(inner_log)
    if total_steps in log_steps:
        outer_logs = jax.tree_map(jnp.mean, outer_log)
        inner_logs = jax.tree_map(lambda x: jnp.mean(x), inner_log)
        monitor_log = dict(
            iteration=iteration,
            total_steps=total_steps,
            reward_per_timestep=jnp.mean(rewards),
            total_reward=total_reward,
            entropy=inner_logs.entropy.mean(),
            pi_loss=inner_logs.pi_loss.mean(),
            baseline_loss=inner_logs.baseline_loss.mean(),
            inner_grad_norm=inner_logs.grad_norm.mean(),
            outer_grad_norm=outer_logs.grad_norm.mean(),
            ent_coef=inner_logs.ent_coef.mean(),
            learning_rate=inner_logs.learning_rate.mean(),
            eta_output=inner_logs.eta_output.mean(),
            theta_nan=theta_nan,
            eta_nan=eta_nan,
        )
        if config['opt_type'] == 'adam' and config['eta_net'] == 'simple':
            count, mu, nu = jax.tree_flatten(eta_opt_state[-1])[0]
            adam_log = dict(
                count=jnp.mean(count),
                mu=jnp.mean(mu),
                nu=jnp.mean(nu),
            )
            monitor_log.update(adam_log)
        reporter(**monitor_log)
    if total_steps in eval_steps:
        path = os.path.join(reporter.logdir, f"checkpoint_{total_steps}.ckpt")
        theta = jax.device_get(theta)
        eta = jax.device_get(eta)
        eta_opt_state = jax.device_get(eta_opt_state)
        with open(path, 'wb') as checkpoint_file:
            pickle.dump((theta, eta, eta_opt_state), checkpoint_file)
    if theta_nan or eta_nan:
        path = os.path.join(reporter.logdir, f"nan_checkpoint_{total_steps}.ckpt")
        has_previous = any(['nan_checkpoint_' in x for x in glob.glob(os.path.join(reporter.logdir, "*"))])
        if not has_previous:
            theta = jax.device_get(theta)
            eta = jax.device_get(eta)
            eta_opt_state = jax.device_get(eta_opt_state)
            with open(path, 'wb') as checkpoint_file:
                pickle.dump((theta, eta, eta_opt_state), checkpoint_file)

    return logdict


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
    k1, k2 = jrandom.split(k1)
    total_reward = 0.0
    total_steps = 0
    t = 0
    logdict = collections.defaultdict(list)
    steps_per_iter = (
        config['H']
        * config['inner_batch_size']
        * config['outer_batch_size']
        * config['truncation_length']
    )
    log_steps = set([
        i // steps_per_iter * steps_per_iter
        for i in range(0, config['stop_steps'],
                       config['log_interval_steps'])])
    eval_steps = set([
        i // steps_per_iter * steps_per_iter
        for i in range(0, config['stop_steps'],
                       config['eval_interval_steps'])])
    gs = []
    while total_steps < config['stop_steps']:
        k1, k2 = jrandom.split(k1)
        outputs = train_step_(k2, eta, theta, mean_rewards, ts, eta_opt_state)
        (eta, theta, rewards, mean_rewards, ts, eta_opt_state,
            outer_log, inner_log, g, theta_nan, eta_nan, old_theta, old_eta,
            old_eta_opt_state) = outputs
        total_reward += rewards.sum()
        total_steps += rewards.size
        t += 1
        logdict = log_status(
            reporter=reporter,
            logdict=logdict,
            config=config,
            log_steps=log_steps,
            eval_steps=eval_steps,
            iteration=t,
            total_reward=total_reward,
            total_steps=total_steps,
            rewards=rewards,
            inner_log=inner_log,
            outer_log=outer_log,
            eta_opt_state=old_eta_opt_state,
            theta_nan=theta_nan,
            eta_nan=eta_nan,
            theta=old_theta,
            eta=old_eta,
        )
        if config['save_gradients']:
            gs.append(g)
    if config['save_gradients']:
        gg = jnp.array([jax.tree_flatten(g)[0] for g in gs])
        path = os.path.join(reporter.logdir, f"gradients.csv")
        pd.DataFrame(dict(
            grad=gg.reshape(-1).tolist(),
        )).to_csv(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='ray')
    args = parser.parse_args()

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
        'reward_flip_interval': 102_400,
        'randomize_lt': True,
        'reset_theta_at_reward_flip': False,
        'H': 16,
        'seed': 0,
        'inner_batch_size': 5,
        'outer_batch_size': 1,
        'gamma': 0.99,

        'inner_loop_fn': 'separate_value',
        'inner_loss': 'no_is',
        'outer_loss': 'first_state',
        'meta_lambda': 0.0,
        'truncation_length': 16,
        'auc_loss': True,
        'eta_context_len': 10,
        'vf_coef': 1.0,
        'ent_coef': 0.0,

        'train_step': 'mg',
        'fd_epsilon': 0.01,
        'save_gradients': False,

        'theta_net': 'actor_net',

        'eta_net': 'context',
        'eta_net_kwargs': {
            'num_outputs': 1,
        },

        'inner_weight_decay': 0.0,
        'inner_opt_type': 'sgd',
        'inner_opt_kwargs': {
            'learning_rate': 0.1,
        },
        'max_inner_grad_norm': 0.0,

        'outer_weight_decay': 0.0,
        'opt_type': 'adam',
        'opt_kwargs': {
            'learning_rate': 1e-4,
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1E-4,
        },
        'max_outer_grad_norm': 0.0,

        'log_interval_steps': 100,
        'stop_steps': 20_000_000,
        'eval_interval_steps': 100_000_000,
        'eval_steps': 5_000_000,
    }
    if args.mode == 'ray':
        analysis = tune.run(
            trainable,
            name='simple_maze:test_single_lifetime:3',
            config=config,
            resources_per_trial={
                'cpu': 3.0,
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
        light_config = {
        }
        config.update(light_config)
        _, config = next(tune.suggest.variant_generator.generate_variants(config))
        with jax.disable_jit():
            trainable(config, reporter=lambda **x: x)
