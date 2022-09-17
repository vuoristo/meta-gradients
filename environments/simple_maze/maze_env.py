from typing import Union, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex

Array = chex.Array
Scalar = chex.Scalar
Numeric = chex.Numeric

def get_maze_env(
    side: Scalar,
    episodic: bool,
    fixed_length_episodes: bool=False,
    episode_max_len: int=10,
    reward_noise_std: float=0.0,
) -> Any:
    def get_free_loc(
        rngkey: Array,
        state: Array,
    ) -> Array:
        reserved = state[0] + state[1] + state[2]
        free = jnp.where(reserved == 0, size=side * side)[0]
        index = jrandom.randint(rngkey, (1,), 0, jnp.sum(1 - reserved))[0]
        return free[index]

    def get_obs(
        state: Array,
        time: Array,
    ) -> Array:
        x_index = jnp.arange(side*side) // side
        y_index = jnp.arange(side*side) % side
        player_coords = jax.nn.one_hot([jnp.sum(state[0] * x_index),
                                        jnp.sum(state[0] * y_index)], side)
        green_coords = jax.nn.one_hot([jnp.sum(state[1] * x_index),
                                       jnp.sum(state[1] * y_index)], side)
        blue_coords = jax.nn.one_hot([jnp.sum(state[2] * x_index),
                                      jnp.sum(state[2] * y_index)], side)
        obs = jnp.concatenate([player_coords, green_coords, blue_coords], axis=0).flatten()
        if fixed_length_episodes:
            obs = jnp.concatenate([obs, jax.nn.one_hot(time, episode_max_len).flatten()], axis=0)
        return obs

    def get_state(
        obs: Array,
    ) -> Array:
        x_index = jnp.arange(side) * side
        y_index = jnp.arange(side)
        state_ = obs[:6*side].reshape((6, side))
        player_coords = jax.nn.one_hot(jnp.sum(state_[0] * x_index)
                                       + jnp.sum(state_[1] * y_index), side*side)
        green_coords = jax.nn.one_hot(jnp.sum(state_[2] * x_index)
                                     + jnp.sum(state_[3] * y_index), side*side)
        blue_coords = jax.nn.one_hot(jnp.sum(state_[4] * x_index)
                                     + jnp.sum(state_[5] * y_index), side*side)
        state = jnp.stack([player_coords, green_coords, blue_coords])
        if fixed_length_episodes:
            time = jnp.argmax(obs[6*side:])
        else:
            time = jnp.zeros(1, dtype=jnp.int32)
        return state, time

    def reset_maze(
        rngkey: Array,
    ) -> Array:
        state = jnp.zeros((3, side*side), dtype=jnp.int32)
        time = jnp.zeros(1, dtype=jnp.int32)
        k1, k2 = jrandom.split(rngkey)
        player_loc = get_free_loc(k2, state)
        state = state.at[0, player_loc].set(1)
        k1, k2 = jrandom.split(k1)
        green_loc = get_free_loc(k2, state)
        state = state.at[1, green_loc].set(1)
        k1, k2 = jrandom.split(k1)
        blue_loc = get_free_loc(k2, state)
        state = state.at[2, blue_loc].set(1)
        return get_obs(state, time)

    def step_maze(
        rngkey: Array,
        obs: Array,
        action: Array,
        rewards: Array,
    ) -> Union[Array, Scalar, Scalar]:
        k1, k2 = jrandom.split(rngkey)
        state, time = get_state(obs)
        a_to_move = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        obs_ = obs[:6*side].reshape(6, side)
        player_loc = jnp.concatenate([jnp.where(obs_[0] == 1, size=1)[0],
                                      jnp.where(obs_[1] == 1, size=1)[0]])
        new_player_loc = jnp.clip(player_loc + a_to_move[action], 0, side-1)
        new_player_state = jax.nn.one_hot(
            new_player_loc[0] * side + new_player_loc[1], side*side)
        new_state = state.at[0].set(new_player_state)
        green_hit = jnp.all(state[1] == new_state[0])
        rnd_green_loc = get_free_loc(k2, new_state)
        new_green_state = jax.nn.one_hot(rnd_green_loc, side*side)
        new_green_state = jax.lax.select(green_hit, new_green_state, new_state[1])
        new_state = new_state.at[1].set(new_green_state)
        blue_hit = jnp.all(state[2] == new_state[0])
        k1, k2 = jrandom.split(k1)
        rnd_blue_loc = get_free_loc(k2, new_state)
        new_blue_state = jax.nn.one_hot(rnd_blue_loc, side*side)
        new_blue_state = jax.lax.select(blue_hit, new_blue_state, new_state[2])
        new_state = new_state.at[2].set(new_blue_state)
        k1, k2 = jrandom.split(k1)
        reward_noise = jrandom.normal(k2) * reward_noise_std
        reward = (-0.04
                  + blue_hit * (rewards[0] + 0.04)
                  + green_hit * (rewards[1] + 0.04)) + reward_noise
        terminate = 0.0
        if episodic and not fixed_length_episodes:
            terminate = jnp.array(blue_hit | green_hit, dtype=reward.dtype)
        elif episodic and fixed_length_episodes:
            time = time + 1
            terminate = jnp.array(time == episode_max_len, dtype=reward.dtype)
        obs = get_obs(new_state, time)
        return obs, reward, terminate

    def resample_goals(rngkey):
        g = jnp.array([[-1.0, 1.0], [1.0, -1.0]])
        goals = g[jrandom.randint(rngkey, (1,), minval=0, maxval=2)][0]
        return goals

    return reset_maze, step_maze, resample_goals


if __name__ == '__main__':
    reset, step, resample_goals_fn = get_maze_env(3, episodic=True, fixed_length_episodes=False)
    k1 = jrandom.PRNGKey(0)
    k1, k2 = jrandom.split(k1)
    obs = reset(k2)
    rewards = resample_goals_fn(k2)
    print(obs)
    for i in range(100):
        k1, k2 = jrandom.split(k1)
        action = jrandom.randint(k2, (), 0, 4)
        k1, k2 = jrandom.split(k1)
        obs, r, done = step(k2, obs, action, rewards)
        print(obs[:18].reshape(6, 3))
        print(r)
        print(action)
