from pathlib import Path

import pandas as pd
import pytest

import navis


@pytest.fixture(scope="session")
def data_dir():
    return Path(__file__).resolve().parent.parent / "navis" / "data"


@pytest.fixture(
    params=["Path", "pathstr", "swcstr", "textbuffer", "rawbuffer", "DataFrame"],
)
def swc_source(request, swc_paths):
    swc_path = swc_paths[0]
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
def swc_source_multi(request, swc_paths):
    fpath = swc_paths[0]
    dpath = fpath.parent
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


def data_paths(dpath, glob="*"):
    return sorted(dpath.glob(glob))


@pytest.fixture(scope="session")
def swc_paths(data_dir: Path):
    return data_paths(data_dir / "swc", "*.swc")


@pytest.fixture(scope="session")
def gml_paths(data_dir: Path):
    return data_paths(data_dir / "gml", "*.gml")


@pytest.fixture(scope="session")
def obj_paths(data_dir: Path):
    return data_paths(data_dir / "obj", "*.obj")


@pytest.fixture(scope="session")
def synapses_paths(data_dir: Path):
    return data_paths(data_dir / "synapses", "*.csv")


@pytest.fixture(scope="session")
def volumes_paths(data_dir: Path):
    return data_paths(data_dir / "volumes", "*.obj")


@pytest.fixture
def neurons(swc_paths, synapses_paths):
    swc_reader = navis.io.swc_io.SwcReader()
    out = []
    for swc_path, syn_path in zip(swc_paths, synapses_paths):
        neuron = swc_reader.read_file_path(swc_path)
        neuron.connectors = pd.read_csv(syn_path)
        out.append(neuron)
    return out
