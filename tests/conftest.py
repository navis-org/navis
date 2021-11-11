from pathlib import Path
import os

import pandas as pd
import pytest
import numpy as np
import nrrd

import navis


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent.parent / "navis" / "data"


@pytest.fixture(
    params=["Path", "pathstr", "swcstr", "textbuffer", "rawbuffer", "DataFrame"]
)
def swc_source(request, data_dir: Path):
    swc_path: Path = data_dir / "swc" / "722817260.swc"
    if request.param == "Path":
        yield swc_path
    elif request.param == "pathstr":
        yield str(swc_path)
    elif request.param == "swcstr":
        yield swc_path.read_text()
    elif request.param == "textbuffer":
        with open(swc_path) as f:
            yield f
    elif request.param == "rawbuffer":
        with open(swc_path, "rb") as f:
            yield f
    elif request.param == "DataFrame":
        df = pd.read_csv(swc_path, " ", header=None, comment="#")
        df.columns = navis.io.swc_io.NODE_COLUMNS
        yield df
    else:
        raise ValueError("Unknown parameter")


@pytest.fixture(
    params=["dirstr", "dirpath", "list", "listwithdir"],
)
def swc_source_multi(request, data_dir: Path):
    dpath = data_dir / "swc"
    fpath = dpath / "722817260.swc"
    if request.param == "dirstr":
        yield str(dpath)
    elif request.param == "dirpath":
        yield dpath
    elif request.param == "list":
        yield [fpath, fpath]
    elif request.param == "listwithdir":
        yield [dpath, fpath]
    else:
        raise ValueError(f"Unknown parameter '{request.param}'")


@pytest.fixture
def voxel_nrrd_path(tmp_path):
    parent = tmp_path / "nrrd"
    parent.mkdir()
    path = parent / "simple.nrrd"
    data = np.zeros((15, 15, 15))
    rng = np.random.RandomState(1991)
    core = rng.random((5, 5, 15))
    data[5:10, 5:10, :] = core

    header = {
        "space directions": np.diag([1, 2, 3]).tolist(),
        "space units": ["um", "um", "um"],
    }
    nrrd.write(os.fspath(path), data, header)

    return path
