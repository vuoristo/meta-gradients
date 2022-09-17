import tree
import numpy as np
import jax
import jax.numpy as jnp


def pack_namedtuple(xs, axis=0):
    return tree.map_structure(lambda *xs: np.stack(xs, axis=axis), *xs)


def pack_namedtuple_jnp(xs, axis=0):
    return jax.tree_util.tree_multimap(lambda *xs: jnp.stack(xs, axis=axis), *xs)


def unpack_namedtuple_jnp(structure, axis=0):
    transposed = tree.map_structure(lambda t: jnp.moveaxis(t, axis, 0), structure)
    flat = tree.flatten(transposed)
    unpacked = list(map(lambda xs: tree.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def unpack_namedtuple_onp(structure, axis=0):
    transposed = tree.map_structure(lambda t: np.moveaxis(t, axis, 0), structure)
    flat = tree.flatten(transposed)
    unpacked = list(map(lambda xs: tree.unflatten_as(structure, xs), zip(*flat)))
    return unpacked


def tree_sum(tree):
    """Compute the sum of a pytree of arrays."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return sum(jnp.sum(x) for x in leaves)


def clip_grads_return_norm(grad_tree, max_norm):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = jax.experimental.optimizers.l2_norm(grad_tree)
    normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
    return jax.tree_util.tree_map(normalize, grad_tree), norm
