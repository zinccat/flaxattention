import inspect
import jax
from typing import Callable, Tuple, Optional, Union, List
from enum import Enum
from jax import Array

_score_mod_signature = Callable[[Array, Array, Array, Array, Array], Array]
_mask_mod_signature = Callable[[Array, Array, Array, Array], Array]

class _ModificationType(Enum):
    """Enum for the type of modification function.
    - SCORE_MOD: score_mod function which accepts a score as the first argument
    - mask_mod: mask function which does not accept a score and is only used for generating
    block mask
    """

    SCORE_MOD = 1
    MASK_MOD = 2
    UNKNOWN = 3

def _get_mod_type(fn: Callable) -> _ModificationType:
    """Get the type of modification function.
    This function inspects the number of positional arguments of the function to determine
    the type of modification function. If the function has 5 positional arguments, it is
    considered as a score_mod function. If the function has 4 positional arguments, it is
    considered as a mask function.
    """
    num_positional_args = sum(
        1
        for param in inspect.signature(fn).parameters.values()
        if param.default == inspect.Parameter.empty
    )
    assert num_positional_args == 5 or num_positional_args == 4
    if num_positional_args == 5:
        return _ModificationType.SCORE_MOD
    elif num_positional_args == 4:
        return _ModificationType.MASK_MOD
    else:
        return _ModificationType.UNKNOWN
    
def _vmap_for_bhqkv(
    fn: Callable,
    prefix: Tuple[Optional[int], ...],
    suffix: Tuple[Optional[int], ...] = (),
    out_axes: Union[int, Tuple[Optional[int], ...]] = 0,
    group_dim: bool = False,
) -> Callable:
    """
    Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimensional inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (Callable): The function to vmap.
        prefix (Tuple): The prefix of the vmap. For score_mod functions, this should be set to (0,). For mask_mods, use ().
        suffix (Tuple): Additional None entries for other buffers or arguments.
        out_axes (Union[int, Tuple[Optional[int], ...]]): The output axes for the vmapped function.
        group_dim (bool): Whether to include the group dimension.

    Returns:
        Callable: The vmapped function.
    """
    dimensions: List[Tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),  # Map over kv_idx
        (None, None, 0, None),  # Map over q_idx
        (None, 0, None, None),  # Map over h
    ]

    if group_dim:
        dimensions += [
            (None, 0, None, None),  # Map over group dimension
        ]

    dimensions += [
        (0, None, None, None),  # Map over batch dimension
    ]

    for dims in dimensions:
        in_axes = prefix + dims + suffix
        fn = jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
    return fn

def _vmap_for_qkv(
    fn: Callable,
    prefix: Tuple[Optional[int], ...],
    suffix: Tuple[Optional[int], ...] = (),
    out_axes: Union[int, Tuple[Optional[int], ...]] = 0,
    group_dim: bool = False,
) -> Callable:
    """
    Used to vmap both score_mods and mask_mods over 4-dimensional/5-dimensional inputs.
    Mapping over the [b, hq, q_idx, kv_idx] or [b, hkv, g, q_idx, kv_idx] dimensions.

    Args:
        fn (Callable): The function to vmap.
        prefix (Tuple): The prefix of the vmap. For score_mod functions, this should be set to (0,). For mask_mods, use ().
        suffix (Tuple): Additional None entries for other buffers or arguments.
        out_axes (Union[int, Tuple[Optional[int], ...]]): The output axes for the vmapped function.
        group_dim (bool): Whether to include the group dimension.

    Returns:
        Callable: The vmapped function.
    """
    dimensions: List[Tuple[None | int, None | int, None | int, None | int]] = []
    dimensions = [
        (None, None, None, 0),  # Map over kv_idx
        (None, None, 0, None),  # Map over q_idx
    ]

    for dims in dimensions:
        in_axes = prefix + dims + suffix
        fn = jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
    return fn