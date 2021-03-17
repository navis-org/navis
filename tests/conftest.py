from pathlib import Path

import pandas as pd
import pytest

import navis


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent.parent / "navis" / "data"


@pytest.fixture(params=["Path", "pathstr", "swcstr", "textbuffer", "rawbuffer", "DataFrame"])
def swc_source(request, data_dir: Path):
    swc_path: Path = data_dir / "swc" / "722817260.swc"
    if request.param == "Path":
        yield swc_path
    elif request.param == "path-str":
        yield str(swc_path)
    elif request.param == "swc-str":
        yield swc_path.read_text()
    elif request.param == "text-buffer":
        with open(swc_path) as f:
            yield f
    elif request.param == "bin-buffer":
        with open(swc_path, "rb") as f:
            yield f
    elif request.param == "DataFrame":
        df = pd.read_csv(swc_path, " ", header=None, comment="#")
        df.columns = navis.io.swc_io.NODE_COLUMNS
        yield df
    else:
        raise ValueError("Unknown parameter")
