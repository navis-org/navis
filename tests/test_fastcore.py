import navis
import pytest

import numpy as np

from pathlib import Path


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("weight", ["weight", None])
@pytest.mark.parametrize("limit", [np.inf, 100])
@pytest.mark.parametrize("from_", [None, np.arange(1, 100)])
def test_geodesic_matrix(directed, weight, limit, from_):
    n = navis.example_neurons(1, kind="skeleton")

    # Make sure that the fastcore package is installed (otherwise this test is useless)
    if navis.utils.fastcore is None:
        return

    # Save fastcore
    fastcore = navis.utils.fastcore

    # Compute the geodesic matrix with fastcore
    m_with = navis.geodesic_matrix(
        n, directed=directed, weight=weight, limit=limit, from_=from_
    )

    # Compute without
    try:
        navis.utils.fastcore = None
        m_without = navis.geodesic_matrix(
            n, directed=directed, weight=weight, limit=limit, from_=from_
        )
    except:
        raise
    finally:
        navis.utils.fastcore = fastcore

    assert np.allclose(m_with, m_without)


@pytest.mark.parametrize("recursive", [True, False])
def test_prune_twigs(recursive):
    n = navis.example_neurons(1, kind="skeleton")

    # Make sure that the fastcore package is installed (otherwise this test is useless)
    if navis.utils.fastcore is None:
        return

    # Save fastcore
    fastcore = navis.utils.fastcore

    # Prune with fastcore
    n_with = navis.prune_twigs(n, size=5000 / 8, recursive=recursive)

    # Prune without fastcore
    try:
        navis.utils.fastcore = None
        n_without = navis.prune_twigs(n, size=5000 / 8, recursive=recursive)
    except:
        raise
    finally:
        navis.utils.fastcore = fastcore

    assert n_with.n_nodes == n_without.n_nodes


@pytest.mark.parametrize("mode", ["sum", "centrifugal", "centripetal"])
def test_synapse_flow_centrality(mode):
    n = navis.example_neurons(1, kind="skeleton")

    # Make sure that the fastcore package is installed (otherwise this test is useless)
    if navis.utils.fastcore is None:
        return

    # Save fastcore
    fastcore = navis.utils.fastcore

    # Compute flow with fastcore
    sfc_with = navis.synapse_flow_centrality(
        n, mode=mode
    ).nodes.synapse_flow_centrality.values

    # Compute flow without fastcore
    try:
        navis.utils.fastcore = None
        sfc_without = navis.synapse_flow_centrality(
            n, mode=mode
        ).nodes.synapse_flow_centrality.values
    except:
        raise
    finally:
        navis.utils.fastcore = fastcore

    assert np.allclose(sfc_with, sfc_without)


def test_parent_dist():
    n = navis.example_neurons(1, kind="skeleton")

    # Make sure that the fastcore package is installed (otherwise this test is useless)
    if navis.utils.fastcore is None:
        return

    # Save fastcore
    fastcore = navis.utils.fastcore

    # Compute parent dist with fastcore
    pd_with = navis.morpho.mmetrics.parent_dist(n, root_dist=0)

    # Compute parent dist without fastcore
    try:
        navis.utils.fastcore = None
        pd_without = navis.morpho.mmetrics.parent_dist(n, root_dist=0)
    except:
        raise
    finally:
        navis.utils.fastcore = fastcore

    assert np.allclose(pd_with, pd_without)


@pytest.mark.parametrize("min_twig_size", [None, 2])
@pytest.mark.parametrize("to_ignore", [[], np.array([465, 548], dtype=int)])
@pytest.mark.parametrize("method", ["standard", "greedy"])
def test_strahler_index(method, to_ignore, min_twig_size):
    n = navis.example_neurons(1, kind="skeleton")

    # Make sure that the fastcore package is installed (otherwise this test is useless)
    if navis.utils.fastcore is None:
        return

    # Save fastcore
    fastcore = navis.utils.fastcore

    # Compute strahler index with fastcore
    si_with = navis.strahler_index(
        n, to_ignore=to_ignore, method=method, min_twig_size=min_twig_size
    ).nodes.strahler_index.values

    # Compute strahler index without fastcore
    try:
        navis.utils.fastcore = None
        si_without = navis.strahler_index(
            n, to_ignore=to_ignore, method=method, min_twig_size=min_twig_size
        ).nodes.strahler_index.values
    except:
        raise
    finally:
        navis.utils.fastcore = fastcore

    assert np.allclose(si_with, si_without)
