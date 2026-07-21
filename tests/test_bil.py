"""Tests for the Brain Image Library interface.

These are deliberately network-free: we only exercise the pure helpers
(the autoindex parser, the URL mapping, the record flattening and the field
validation). Anything that talks to BIL is left to manual testing, in line
with how the other interfaces are handled.

Note this module must stay free of doctests - pytest runs with
`--doctest-modules`.
"""

import pytest

import numpy as np

from navis.interfaces import brain_image_library as bil


# A verbatim nginx autoindex listing, with a couple of extras thrown in:
# a sort link, a subdirectory, a rounded size, an exact size and an escaped
# space in a filename.
AUTOINDEX = """<html>
<head><title>Index of /22/5c/225c37cacfbd897c/</title></head>
<body>
<h1>Index of /22/5c/225c37cacfbd897c/</h1><hr><pre><a href="../">../</a>
<a href="?C=N;O=D">Name</a>
<a href="212083-19/">212083-19/</a>                     31-Mar-2022 15:12       -
<a href="mouseID_212083_01.swc">mouseID_212083_01.swc</a> 31-Mar-2022 15:12  5245516
<a href="rounded.swc">rounded.swc</a>                   01-Apr-2022 09:00     3.4M
<a href="with%20space.swc">with space.swc</a>            01-Apr-2022 09:00      512
</pre><hr></body>
</html>"""

BASE_URL = "https://download.brainimagelibrary.org/22/5c/225c37cacfbd897c/"


@pytest.fixture
def entries():
    return {e["name"]: e for e in bil._parse_autoindex(AUTOINDEX, BASE_URL)}


def test_autoindex_skips_parent_and_sort_links(entries):
    # "../" and nginx's "?C=N;O=D" column-sort links must not show up - crawling
    # the latter would send us round in circles.
    assert set(entries) == {
        "212083-19",
        "mouseID_212083_01.swc",
        "rounded.swc",
        "with space.swc",
    }


def test_autoindex_directories(entries):
    assert entries["212083-19"]["is_dir"] is True
    assert entries["212083-19"]["size"] is None
    assert entries["212083-19"]["url"] == BASE_URL + "212083-19/"


def test_autoindex_files(entries):
    swc = entries["mouseID_212083_01.swc"]
    assert swc["is_dir"] is False
    assert swc["size"] == 5245516
    assert swc["last_modified"] == "31-Mar-2022 15:12"
    assert swc["url"] == BASE_URL + "mouseID_212083_01.swc"


def test_autoindex_rounded_size(entries):
    # nginx's default autoindex reports rounded sizes
    assert entries["rounded.swc"]["size"] == int(3.4 * 1024**2)


def test_autoindex_unquotes_name_but_keeps_url(entries):
    entry = entries["with space.swc"]
    assert entry["name"] == "with space.swc"
    assert entry["url"] == BASE_URL + "with%20space.swc"
    assert entry["size"] == 512


@pytest.mark.parametrize(
    "size,expected",
    [
        ("3.4M", int(3.4 * 1024**2)),
        ("10G", 10 * 1024**3),
        ("10GB", 10 * 1024**3),
        ("512", 512),
        ("-", None),
        (None, None),
        ("", None),
        ("garbage", None),
    ],
)
def test_parse_size(size, expected):
    assert bil._parse_size(size) == expected


@pytest.mark.parametrize(
    "directory,expected",
    [
        (
            "/bil/data/22/5c/225c37cacfbd897c",
            "https://download.brainimagelibrary.org/22/5c/225c37cacfbd897c/",
        ),
        # Some datasets carry an extra sub-directory
        (
            "/bil/data/cd/db/cddb33924966b07f/735467",
            "https://download.brainimagelibrary.org/cd/db/cddb33924966b07f/735467/",
        ),
        # A trailing slash must not produce a double slash
        (
            "/bil/data/22/5c/225c37cacfbd897c/",
            "https://download.brainimagelibrary.org/22/5c/225c37cacfbd897c/",
        ),
    ],
)
def test_bildirectory_to_url(directory, expected):
    assert bil._bildirectory_to_url(directory) == expected


def test_bildirectory_to_url_always_ends_in_slash():
    # The crawler relies on this: without a trailing slash `urljoin` would
    # silently drop the last path segment.
    assert bil._bildirectory_to_url("/bil/data/22/5c/abc").endswith("/")


@pytest.mark.parametrize("directory", ["/some/other/path", "/bil/data/", ""])
def test_bildirectory_to_url_raises(directory):
    with pytest.raises(ValueError):
        bil._bildirectory_to_url(directory)


def make_record(specimens, **kwargs):
    rec = {
        "Submission": {"project": "U19 Huang", "doi": "submission-doi"},
        "Dataset": [
            {
                "bildirectory": "/bil/data/22/5c/225c37cacfbd897c",
                "title": "Some title",
                "doi": "dataset-doi",
                "generalmodality": "cell morphology",
                "technique": "fMOST",
                # BIL returns these as *strings*
                "dataset_size": "0.447271145",
                "number_of_files": "117",
            }
        ],
        "Specimen": specimens,
        "Assets": [{"bildid": "ace-boo-van"}],
    }
    rec.update(kwargs)
    return rec


def test_flatten_collapses_unanimous_specimens():
    # All specimens agree -> a plain scalar, so that `df.species == 'mouse'`
    # still works.
    rec = make_record([{"species": "mouse"}, {"species": "mouse"}])
    row = bil._flatten_record(rec)
    assert row["species"] == "mouse"
    assert row["n_specimens"] == 2


def test_flatten_keeps_conflicting_specimens():
    # Specimens disagree -> a tuple. We must not silently drop information.
    rec = make_record([{"sex": "Female"}, {"sex": "Male"}, {"sex": "Female"}])
    row = bil._flatten_record(rec)
    assert row["sex"] == ("Female", "Male")
    assert row["n_specimens"] == 3


def test_flatten_casts_numbers():
    row = bil._flatten_record(make_record([]))
    assert isinstance(row["dataset_size_gb"], float)
    assert row["dataset_size_gb"] == pytest.approx(0.447271145)
    assert row["number_of_files"] == 117


def test_flatten_bad_numbers_become_nan():
    rec = make_record([])
    rec["Dataset"][0]["dataset_size"] = ""
    rec["Dataset"][0]["number_of_files"] = "not a number"
    row = bil._flatten_record(rec)
    assert np.isnan(row["dataset_size_gb"])
    assert np.isnan(row["number_of_files"])


def test_flatten_namespaces_colliding_keys():
    # `doi` exists on both Dataset and Submission - they must not collide.
    row = bil._flatten_record(make_record([]))
    assert row["doi"] == "dataset-doi"
    assert row["submission.doi"] == "submission-doi"


def test_flatten_derives_url_and_id():
    row = bil._flatten_record(make_record([]))
    assert row["bildid"] == "ace-boo-van"
    assert row["url"] == (
        "https://download.brainimagelibrary.org/22/5c/225c37cacfbd897c/"
    )


def test_flatten_tolerates_missing_divisions():
    # Any division may be absent on any given record - this must never raise.
    row = bil._flatten_record({})
    assert row["bildid"] is None
    assert row["url"] is None
    assert row["n_specimens"] == 0
    assert np.isnan(row["dataset_size_gb"])


def test_check_fields_accepts_known_fields():
    assert bil._check_fields(["species", "generalmodality", "text"]) is None


def test_check_fields_rejects_unknown_field():
    with pytest.raises(ValueError) as exc:
        bil._check_fields(["bogusfield"])

    msg = str(exc.value)
    assert "bogusfield" in msg
    # The error should point the user at what they *can* use
    assert "species" in msg


def test_search_validates_before_hitting_the_network(monkeypatch):
    # Field validation must happen up-front - if it didn't, this would try to
    # talk to BIL and the test would be flaky (or hang offline).
    def boom(*args, **kwargs):
        raise AssertionError("search() hit the network before validating fields")

    monkeypatch.setattr(bil.session, "get", boom)
    monkeypatch.setattr(bil.session, "post", boom)

    with pytest.raises(ValueError):
        bil.search(bogusfield="x")

    with pytest.raises(ValueError):
        bil.search()  # no filters at all
