"""Tests for the built-in NBLAST work-distribution helpers.

`partition_grid` and `find_optimal_partition` only ever read the query/target
*counts*, so we can exercise them with plain integers / lists standing in for
NeuronLists - no neurons, scoring or timing required.
"""

import inspect

import pytest

from navis.nbl.nblast_funcs import (
    partition_grid,
    find_optimal_partition,
    estimate_target_blocks,
    MIN_BLOCKS_PER_CORE,
)


def q_t(nq, nt):
    """Stand-ins for query/target NeuronLists of the given sizes."""
    return list(range(nq)), list(range(nt))


# --------------------------------------------------------------------------- #
# find_optimal_partition (thin wrapper -> one wave, floored at min-per-core)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cores", [2, 3, 4, 5, 7, 8, 9, 11, 16])
def test_partition_is_valid(n_cores):
    """Grid must be splittable and keep every worker busy."""
    q, t = q_t(1000, 1000)
    n_rows, n_cols = find_optimal_partition(n_cores, q, t)

    # Never split into more pieces than we have neurons
    assert 1 <= n_rows <= len(q)
    assert 1 <= n_cols <= len(t)
    # Enough blocks that no core sits idle
    assert n_rows * n_cols >= n_cores


@pytest.mark.parametrize("n_cores", [3, 5, 7, 11, 13])
def test_primes_avoid_degenerate_grid(n_cores):
    """Prime core counts (>2) must not collapse to a 1xN / Nx1 grid.

    With equal-sized query/target sets a prime like 7 used to yield a 1x7 grid
    where every block carried the full query set; allowing more than one block
    per core lets it use a balanced 2x7 instead.
    """
    q, t = q_t(1000, 1000)
    n_rows, n_cols = find_optimal_partition(n_cores, q, t)
    assert min(n_rows, n_cols) > 1


@pytest.mark.parametrize("n_cores", [2, 3, 4, 5, 7, 8, 9, 16])
def test_full_waves_and_min_blocks_per_core(n_cores):
    """Block count is a whole multiple of n_cores and >= the per-core floor."""
    q, t = q_t(1000, 1000)
    n_rows, n_cols = find_optimal_partition(n_cores, q, t)
    n_blocks = n_rows * n_cols
    assert n_blocks % n_cores == 0                      # complete waves
    assert n_blocks >= MIN_BLOCKS_PER_CORE * n_cores    # spare work per core


def test_symmetry_in_query_target_sizes():
    """Swapping q and t swaps the grid dimensions."""
    n_cores = 7
    rows_a, cols_a = find_optimal_partition(n_cores, *q_t(2000, 500))
    rows_b, cols_b = find_optimal_partition(n_cores, *q_t(500, 2000))
    assert (rows_a, cols_a) == (cols_b, rows_b)


def test_more_rows_when_more_queries():
    """The larger set gets split more finely (less data shipped per worker)."""
    n_rows, n_cols = find_optimal_partition(8, *q_t(4000, 500))
    assert n_rows >= n_cols


@pytest.mark.parametrize("nq,nt", [(1, 1000), (1000, 1), (1, 1), (3, 3)])
def test_tiny_sets_do_not_crash(nq, nt):
    """Few queries/targets fall back gracefully instead of over-splitting."""
    n_rows, n_cols = find_optimal_partition(7, *q_t(nq, nt))
    assert 1 <= n_rows <= nq
    assert 1 <= n_cols <= nt


# --------------------------------------------------------------------------- #
# partition_grid (the general core-aware partitioner)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cores", [4, 7, 8, 13])
@pytest.mark.parametrize("target", [1, 10, 100, 2025])
def test_partition_grid_full_waves(n_cores, target):
    """Whatever the target, the grid is a full multiple of n_cores."""
    n_rows, n_cols = partition_grid(n_cores, 1000, 1000, target_blocks=target)
    assert (n_rows * n_cols) % n_cores == 0
    assert n_rows * n_cols >= MIN_BLOCKS_PER_CORE * n_cores


def test_partition_grid_honours_large_target():
    """A large runtime-derived target produces many (aligned) blocks."""
    n_cores = 8
    n_rows, n_cols = partition_grid(n_cores, 1000, 1000, target_blocks=2000)
    n_blocks = n_rows * n_cols
    assert (n_blocks % n_cores) == 0
    # Close to the requested granularity (rounded up to a full wave)
    assert 2000 <= n_blocks <= 2000 + n_cores


def test_partition_grid_minimises_data_for_balanced_case():
    """With equal q/t and a square-friendly core count we get the square grid."""
    # 16 blocks over 8 cores, 1000x1000 -> 4x4 ships the least data
    assert partition_grid(8, 1000, 1000, target_blocks=8) == (4, 4)


def test_partition_grid_caps_at_available_pairs():
    """Can't make more blocks than there are query/target pairs."""
    n_rows, n_cols = partition_grid(8, 3, 3, target_blocks=1000)
    assert n_rows <= 3 and n_cols <= 3
    assert n_rows * n_cols <= 9


def test_partition_grid_serial_shortcut():
    """A single pair (or single core) is one block."""
    assert partition_grid(8, 1, 1, target_blocks=100) == (1, 1)
    assert partition_grid(1, 1000, 1000, target_blocks=100) == (1, 1)


def test_estimate_target_blocks_signature():
    """The runtime estimator takes only q/t/T - no core-count coupling."""
    params = inspect.signature(estimate_target_blocks).parameters
    assert "n_cores" not in params
    assert set(params) == {"q", "t", "T"}
