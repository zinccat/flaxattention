from typing import Callable, Tuple, Optional, Union
import functools
import operator
import jax
import jax.numpy as jnp
from jax import Array

from .common import (
    _score_mod_signature,
    _mask_mod_signature,
    _ModificationType,
    _get_mod_type,
    _vmap_for_bhqkv,
    _vmap_for_qkv
)

_DEFAULT_SPARSE_BLOCK_SIZE = 128
_LARGE_SPARSE_BLOCK_SIZE = 1 << 30


def noop_mask(
    batch: Array,
    head: Array,
    token_q: Array,
    token_kv: Array,
) -> Array:
    """Returns a noop mask_mod"""
    return jnp.ones((), dtype=jnp.bool)


def _ordered_to_dense(num_blocks_in_row: Array, col_indices: Array):
    num_rows = col_indices.shape[-2]
    num_cols = col_indices.shape[-1]
    batch_dims = num_blocks_in_row.shape[:-1]

    def create_dense_one(kv_num_blocks, kv_indices):
        dense_mask = jnp.zeros((num_rows, num_cols + 1), dtype=jnp.int32)

        row_indices = jnp.arange(num_rows, dtype=jnp.int32)[None, :]
        col_range = jnp.arange(num_cols, dtype=jnp.int32)
        index_mask = col_range < kv_num_blocks[None, :]

        # We write to one spot "out of bounds"
        valid_indices = jnp.where(index_mask, kv_indices, num_cols)

        # set the values in 'a' to 1 where the indices are valid
        dense_mask.at[row_indices, valid_indices].set(1)
        return dense_mask[:, :num_cols]

    create_dense_batched = create_dense_one
    for _ in range(len(batch_dims)):
        create_dense_batched = jax.vmap(create_dense_batched, in_axes=(0, 0))

    out = create_dense_batched(num_blocks_in_row, col_indices)
    return out


def _dense_to_ordered(dense_mask) -> Tuple:
    dense_mask = dense_mask.astype(dtype=jnp.int32)
    num_blocks_in_row = dense_mask.sum(axis=-1)
    col_indices = jnp.argsort(dense_mask, axis=-1, descending=True, stable=True)
    return (
        num_blocks_in_row.astype(jnp.int32),
        col_indices.astype(jnp.int32),
    )


def _transpose_ordered(num_blocks_in_row: Array, col_indices: Array):
    dense = _ordered_to_dense(num_blocks_in_row, col_indices)
    return _dense_to_ordered(jnp.matrix_transpose(dense))


class BlockMask:
    kv_num_blocks: Array
    kv_indices: Array
    full_kv_num_blocks: Optional[Array]
    full_kv_indices: Optional[Array]
    q_num_blocks: Optional[Array]
    q_indices: Optional[Array]
    full_q_num_blocks: Optional[Array]
    full_q_indices: Optional[Array]
    BLOCK_SIZE: Tuple[int, int]
    mask_mod: _mask_mod_signature

    def __init__(
        self,
        kv_num_blocks: Array,
        kv_indices: Array,
        full_kv_num_blocks: Optional[Array],
        full_kv_indices: Optional[Array],
        q_num_blocks: Optional[Array],
        q_indices: Optional[Array],
        full_q_num_blocks: Optional[Array],
        full_q_indices: Optional[Array],
        BLOCK_SIZE: Tuple[int, int],
        mask_mod: _mask_mod_signature,
    ):
        if kv_indices.ndim < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")
        assert kv_num_blocks is not None, "kv_num_blocks must be provided"
        assert kv_indices is not None, "kv_indices must be provided"
        assert q_num_blocks is not None, "q_num_blocks must be provided"
        assert q_indices is not None, "q_indices must be provided"
        assert (full_kv_num_blocks is None) == (
            full_kv_indices is None
        ), "full_kv_num_blocks and full_kv_indices must be both provided or omitted"
        assert (full_q_num_blocks is None) == (
            full_q_indices is None
        ), "full_q_num_blocks and full_q_indices must be both provided or omitted"

        self.kv_num_blocks = kv_num_blocks
        self.kv_indices = kv_indices
        self.full_kv_num_blocks = full_kv_num_blocks
        self.full_kv_indices = full_kv_indices
        self.q_num_blocks = q_num_blocks
        self.q_indices = q_indices
        self.full_q_num_blocks = full_q_num_blocks
        self.full_q_indices = full_q_indices
        self.BLOCK_SIZE = BLOCK_SIZE
        self.mask_mod = mask_mod

    @classmethod
    def from_kv_blocks(
        cls,
        kv_num_blocks: Array,
        kv_indices: Array,
        full_kv_num_blocks: Optional[Array] = None,
        full_kv_indices: Optional[Array] = None,
        BLOCK_SIZE: Union[int, Tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE,
        mask_mod: Optional[_mask_mod_signature] = None,
        pallas: bool = False,
    ):
        if kv_indices.ndim < 2:
            raise RuntimeError("BlockMask must have at least 2 dimensions")

        assert (full_kv_num_blocks is None) == (
            full_kv_indices is None
        ), "full_kv_num_blocks and full_kv_indices must be both provided or omitted"

        # Generate q_num_blocks and q_indices
        q_num_blocks, q_indices = _transpose_ordered(kv_num_blocks, kv_indices)
        if full_kv_num_blocks is not None:
            assert full_kv_indices is not None
            full_q_num_blocks, full_q_indices = _transpose_ordered(
                full_kv_num_blocks, full_kv_indices
            )
        else:
            full_q_num_blocks, full_q_indices = None, None

        if isinstance(BLOCK_SIZE, int):
            BLOCK_SIZE = (BLOCK_SIZE, BLOCK_SIZE)

        mask_mod = mask_mod if mask_mod is not None else noop_mask

        if pallas:
            mask_mod = _vmap_for_qkv(mask_mod, prefix=())

        return cls(
            kv_num_blocks=kv_num_blocks,
            kv_indices=kv_indices,
            full_kv_num_blocks=full_kv_num_blocks,
            full_kv_indices=full_kv_indices,
            q_num_blocks=q_num_blocks,
            q_indices=q_indices,
            full_q_num_blocks=full_q_num_blocks,
            full_q_indices=full_q_indices,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=mask_mod,
        )

    def as_tuple(self, flatten: bool = True):
        """
        Returns a tuple of the attributes of the BlockMask.

        Args:
            flatten (bool): If True, it will flatten the tuple of (KV_BLOCK_SIZE, Q_BLOCK_SIZE)
        """
        block_size = (
            (self.BLOCK_SIZE[0], self.BLOCK_SIZE[1]) if flatten else (self.BLOCK_SIZE,)
        )

        return (
            self.kv_num_blocks,
            self.kv_indices,
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.q_num_blocks,
            self.q_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
            *block_size,
            self.mask_mod,
        )
    
    def __str__(self):
        s = f"BlockMask(shape={self.shape}, sparsity={self.sparsity():.2f}%, \n"
        mask_str = self.to_string().strip()
        s += mask_str
        s += "\n)"
        return s

    def __getitem__(self, index) -> "BlockMask":
        """
        Returns a new BlockMask instance by getting the mask for the given index position.

        Args:
            index: Index to apply to all attributes.

        Example Usage:
            .. code-block:: python

                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                block_mask = create_block_mask(causal_mask, 4, 2, 512, 512, device="cuda")
                assert block_mask.kv_num_blocks.shape == (4,2,4)
                assert block_mask.kv_indices.shape == (4,2,4,4)

                # Index on batch dimension
                new_block_mask = block_mask[0]
                assert new_block_mask.kv_num_blocks.shape == (2,4)
                assert new_block_mask.kv_indices.shape == (2,4,4)

                # Index on batch and head dimension
                new_block_mask = block_mask[0, 1]
                assert new_block_mask.kv_num_blocks.shape == (4,)
                assert new_block_mask.kv_indices.shape == (4,4)

                # slicing on batch and head dimension
                new_block_mask = block_mask[0:2, 1:2]
                assert new_block_mask.kv_num_blocks.shape == (2,1,4)
                assert new_block_mask.kv_indices.shape == (2,1,4,4)

                # slicing on batch, head, and query dimension
                new_block_mask = block_mask[0:2, 1:2, torch.tensor([1], dtype=torch.int32)]
                assert new_block_mask.kv_num_blocks.shape == (2,1,1)
                assert new_block_mask.kv_indices.shape == (2,1,1,4)
        """
        new_kv_num_blocks = self.kv_num_blocks[index]
        new_kv_indices = self.kv_indices[index]
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            new_full_kv_num_blocks = self.full_kv_num_blocks[index]
            new_full_kv_indices = self.full_kv_indices[index]
        else:
            new_full_kv_num_blocks = None
            new_full_kv_indices = None
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            BLOCK_SIZE=self.BLOCK_SIZE,
            mask_mod=None,
        )

    def __repr__(self):
        def shape_or_none(x: Optional[Array]):
            return x.shape if x is not None else None

        return (
            f"BlockMask(\n"
            f"    kv_num_blocks={self.kv_num_blocks.shape},\n"
            f"    kv_indices={self.kv_indices.shape},\n"
            f"    full_kv_num_blocks={shape_or_none(self.full_kv_num_blocks )},\n"
            f"    full_kv_indices={shape_or_none(self.full_kv_indices)},\n"
            f"    q_num_blocks={shape_or_none(self.q_num_blocks)},\n"
            f"    q_indices={shape_or_none(self.q_indices)},\n"
            f"    full_q_num_blocks={shape_or_none(self.full_q_num_blocks)},\n"
            f"    full_q_indices={shape_or_none(self.full_q_indices)},\n"
            f"    BLOCK_SIZE={self.BLOCK_SIZE},\n"
            f"    shape={self.shape},\n"
            f"    sparsity={self.sparsity():.2f}%,\n"
            f"    mask_mod={self.mask_mod.__name__ if hasattr(self.mask_mod, '__name__') else self.mask_mod}\n"
            f")"
        )
    
    @property
    def shape(self):
        """Returns the shape of the mask."""
        *batch_dims, q_length, _ = self.kv_indices.shape
        q_length = self.kv_indices.shape[-2] * self.BLOCK_SIZE[0]
        kv_length = self.kv_indices.shape[-1] * self.BLOCK_SIZE[1]
        return tuple(batch_dims + [q_length, kv_length])

    def numel(self):
        """Returns the number of elements (not accounting for sparsity) in the mask."""
        shape = self.shape

        def _prod(xs):
            return functools.reduce(operator.mul, xs, 1)

        return _prod(shape)

    def sparsity(self) -> float:
        """Computes the percentage of blocks that are sparse (i.e. not computed)"""
        total_size = self.numel()
        computed_blocks = self.kv_num_blocks.sum()
        if self.full_kv_num_blocks is not None:
            computed_blocks += self.full_kv_num_blocks.sum()

        computed_size = computed_blocks.item() * self.BLOCK_SIZE[0] * self.BLOCK_SIZE[1]
        dense_ratio = computed_size / total_size
        return 100 * (1 - dense_ratio)
    
    def to_dense(self) -> Array:
        """Returns a dense block that is equivalent to the block mask."""
        partial_dense = _ordered_to_dense(self.kv_num_blocks, self.kv_indices)
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            return partial_dense | _ordered_to_dense(
                self.full_kv_num_blocks, self.full_kv_indices
            )
        return partial_dense


def _create_empty_block_mask(query: Array, key: Array, pallas: bool=False) -> BlockMask:
    r"""Default block mask for flex attention.
    If users don't specify any block sparse mask info, we create this
    empty block sparse mask. Which creates a BlockMask with 1 block that is the full length
    of the query and key Arrays.
    """
    return BlockMask.from_kv_blocks(
        kv_num_blocks=jnp.ones([1, 1, 1], dtype=jnp.int32),
        kv_indices=jnp.zeros([1, 1, 1, 1], dtype=jnp.int32),
        BLOCK_SIZE=_LARGE_SPARSE_BLOCK_SIZE,
        pallas=pallas,
    )


def _create_sparse_block_from_block_mask(
    block_mask: Tuple[Array, Optional[Array]],
    mask_mod: Optional[Callable],
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    partial_blocks, full_blocks = block_mask

    partial_bm = _dense_to_ordered(partial_blocks)
    if full_blocks is not None:
        full_bm = _dense_to_ordered(full_blocks)
    else:
        full_bm = (None, None)

    return BlockMask.from_kv_blocks(
        partial_bm[0],
        partial_bm[1],
        full_bm[0],
        full_bm[1],
        BLOCK_SIZE=(KV_BLOCK_SIZE, Q_BLOCK_SIZE),
        mask_mod=mask_mod,
    )


def create_mask(
    mod_fn: Union[_score_mod_signature, _mask_mod_signature],
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
) -> Array:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (Union[_score_mod_signature, _mask_mod_signature]): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on. (removed for now)

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    if B is None:
        B = 1
    if H is None:
        H = 1
    b = jnp.arange(B)
    h = jnp.arange(H)
    m = jnp.arange(Q_LEN)
    n = jnp.arange(KV_LEN)

    mod_type = _get_mod_type(mod_fn)
    if mod_type == _ModificationType.SCORE_MOD:
        # run on an empty tensor to get the mask (we only care about neginf values)
        score_mod = mod_fn
        score_mod = _vmap_for_bhqkv(score_mod, prefix=(0,))  # first input is score
        out = score_mod(jnp.zeros((B, H, Q_LEN, KV_LEN)), b, h, m, n)
        mask = jnp.where(
            jnp.isneginf(out), False, True
        )  # mask is True where the score is not -inf, False ones are masked
        return mask
    elif mod_type == _ModificationType.MASK_MOD:
        mask_mod = mod_fn
        mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
        mask = mask_mod(b, h, m, n)
        return mask
    else:
        raise ValueError("Unknown modification function type")


def _broadcast_to_dim(x, dim):
    while x.ndim < dim:
        x = x[None, :]
    return x


def _round_up_to_multiple(x, multiple):
    return (x + multiple - 1) // multiple * multiple


def _convert_mask_to_block_mask(
    mask: Array,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    separate_full_blocks: bool = False,
) -> Tuple[Array, Optional[Array]]:
    """
    Convert a mask tensor to a block mask tensor.
    """
    assert mask.dtype == jnp.bool
    mask = _broadcast_to_dim(mask, 4)
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.reshape(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.transpose(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask_block_sum = mask.sum(
        axis=[-2, -1]
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    if separate_full_blocks:
        # full blocks are blocks that are fully masked, separating them allows for not computing them
        full_block_sum = Q_BLOCK_SIZE * KV_BLOCK_SIZE
        full_blocks = mask_block_sum == full_block_sum
        partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        partial_blocks = partial_blocks.astype(dtype=jnp.int8)
        full_blocks = full_blocks.astype(dtype=jnp.int8)
        return partial_blocks, full_blocks
    else:
        partial_blocks = mask_block_sum > 0
        partial_blocks = partial_blocks.astype(dtype=jnp.int8)
        return partial_blocks, None


def or_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the union of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def or_mask(b, h, q_idx, kv_idx):
        result = jnp.zeros((), dtype=jnp.bool)
        for mask in mask_mods:
            result = result | mask(b, h, q_idx, kv_idx)
        return result

    return or_mask


def and_masks(*mask_mods: _mask_mod_signature) -> _mask_mod_signature:
    """Returns a mask_mod that's the intersection of provided mask_mods"""
    if not all(callable(arg) for arg in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def and_mask(b, h, q_idx, kv_idx):
        result = jnp.ones((), dtype=jnp.bool)
        for mask in mask_mods:
            result = result & mask(b, h, q_idx, kv_idx)
        return result

    return and_mask


def _create_block_mask_inner(
    mask_mod: Callable,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    KV_BLOCK_SIZE: int,
    Q_BLOCK_SIZE: int,
):
    mask_Array = create_mask(mask_mod, B, H, Q_LEN, KV_LEN)
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        mask_Array,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        separate_full_blocks=True,
    )
    return partial_block_mask, full_block_mask


def create_block_mask(
    mask_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: Union[int, Tuple[int, int]] = _DEFAULT_SPARSE_BLOCK_SIZE,
    _compile=False,
) -> BlockMask:
    inner_func = _create_block_mask_inner

    if B is None:
        B = 1
    if H is None:
        H = 1
    if isinstance(BLOCK_SIZE, int):
        Q_BLOCK_SIZE = BLOCK_SIZE
        KV_BLOCK_SIZE = BLOCK_SIZE
    else:
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

    if Q_LEN < 128:
        Q_BLOCK_SIZE = Q_LEN
    else:
        Q_LEN = _round_up_to_multiple(Q_LEN, Q_BLOCK_SIZE)
    KV_LEN = _round_up_to_multiple(KV_LEN, KV_BLOCK_SIZE)
    if _compile:
        inner_func = jax.jit(
            inner_func, static_argnums=(0, 1, 2, 3, 4, 5, 6)
        )  # this is too weird, there might be a better way to do this
    partial_block_mask, full_block_mask = inner_func(
        mask_mod, B, H, Q_LEN, KV_LEN, KV_BLOCK_SIZE, Q_BLOCK_SIZE
    )
    block_mask = _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask), mask_mod
    )
    return block_mask
