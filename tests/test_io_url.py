"""Tests for reading neurons from URLs.

These use a local HTTP server (see the `http_server` fixture in conftest.py) and
so do not touch the actual network.

Historically the parallel URL path handed the *downloaded bytes* - rather than
the URL - to the reader. That meant `parse_filename` never ran, so neurons came
back without `name`/`file`/`origin`, with `fmt` silently ignored, and reading
meshes failed outright. The tests below pin the invariant that was violated:
reading in parallel must produce the same neurons as reading serially.

Note this module must stay free of doctests - pytest runs with
`--doctest-modules`.
"""

import navis
import pytest
import threading

from pathlib import Path

from navis.io import base


# Covers: the default, both "force parallel" spellings, serial, and an explicit
# threshold. Note we ship 5 example SWCs and `PARALLEL_THRESHOLD_URL` is 5, so
# even "auto" takes the parallel path here.
PARALLEL = ["auto", True, False, 2, ("auto", 1)]


@pytest.mark.parametrize("parallel", PARALLEL)
def test_read_swc_url_matches_serial(swc_urls, parallel):
    """Reading in parallel must give the same neurons as reading serially."""
    serial = navis.read_swc(swc_urls, parallel=False)
    par = navis.read_swc(swc_urls, parallel=parallel)

    assert len(par) == len(swc_urls)

    # Order is preserved and provenance survives
    assert [n.origin for n in par] == swc_urls
    assert [n.name for n in par] == [Path(u).stem for u in swc_urls]
    assert [n.file for n in par] == [Path(u).name for u in swc_urls]

    for a, b in zip(serial, par):
        assert a.name == b.name
        assert a.origin == b.origin
        assert a.file == b.file
        assert a.n_nodes == b.n_nodes

    # N.B. do NOT compare `.id`: with the default `fmt` it's a fresh uuid4 on
    # every read, so it differs between two *correct* reads.


@pytest.mark.parametrize("parallel", PARALLEL)
def test_read_swc_url_honours_fmt(swc_urls, parallel):
    """`fmt` used to be silently ignored when reading URLs in parallel."""
    nl = navis.read_swc(swc_urls, parallel=parallel, fmt="{id}.swc")

    assert [str(n.id) for n in nl] == [Path(u).stem for u in swc_urls]


@pytest.mark.parametrize("parallel", PARALLEL)
def test_read_mesh_url(obj_urls, obj_paths, parallel):
    """Reading meshes from URLs in parallel used to raise (missing `file` attr)."""
    nl = navis.read_mesh(obj_urls, parallel=parallel)

    assert len(nl) == len(obj_urls)
    assert [n.name for n in nl] == [Path(u).stem for u in obj_urls]

    local = navis.read_mesh([str(p) for p in obj_paths], parallel=False)
    assert [n.n_vertices for n in nl] == [n.n_vertices for n in local]


def test_read_swc_url_404_raises(http_server):
    with pytest.raises(Exception):
        navis.read_swc([f"{http_server}/swc/does_not_exist.swc"], parallel=True)


# ---------------------------------------------------------------------------
# Socket-free tests for `parallel_read` itself. These pin the two bugs directly
# and run in milliseconds.
# ---------------------------------------------------------------------------


def make_urls(n):
    return [f"http://127.0.0.1/{i}.swc" for i in range(n)]


def test_parallel_read_passes_urls_not_bytes():
    """The URL - not the downloaded content - must reach the reader function."""
    seen = []

    def read_fn(obj):
        seen.append((obj, threading.current_thread().name))
        return obj

    urls = make_urls(base.PARALLEL_THRESHOLD_URL)
    out = base.parallel_read(read_fn, urls, "auto")

    # `.map` returns in input order even though the threads finish out of order
    assert out == urls

    # The reader saw the URLs themselves - never the downloaded bytes
    assert {o for o, _ in seen} == set(urls)
    assert not any(isinstance(o, bytes) for o, _ in seen)

    # ... and we actually went parallel (this is what the threshold change buys)
    assert any(name != "MainThread" for _, name in seen)


def test_parallel_read_urls_below_threshold_are_serial():
    threads = []

    def read_fn(obj):
        threads.append(threading.current_thread().name)
        return obj

    urls = make_urls(base.PARALLEL_THRESHOLD_URL - 1)
    base.parallel_read(read_fn, urls, "auto")

    assert set(threads) == {"MainThread"}


def test_parallel_read_empty_spawns_nothing():
    assert base.parallel_read(lambda obj: obj, [], parallel=True) == []


def test_parallel_read_does_not_consume_generator():
    urls = make_urls(base.PARALLEL_THRESHOLD_URL)

    assert base.parallel_read(lambda obj: obj, (u for u in urls), "auto") == urls
