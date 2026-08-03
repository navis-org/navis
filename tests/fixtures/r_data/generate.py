"""Regenerate `objects.rda`.

Run with `python tests/fixtures/r_data/generate.py` from the repo root. Needs
`rdata` >= 1.0 (i.e. Python >= 3.11) since that is what can write R data files.

The fixture exists so that the *reading* side of `navis.io.rda_io` is covered
on Python 3.10, where `rdata` is too old to write and every roundtrip test is
skipped. Neurons are pruned and resampled to keep the file small while still
carrying nodes, connectors, dotprops and a mesh.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    import navis

    navis.set_pbars(hide=True)

    nl = navis.example_neurons(2, kind="skeleton")
    nl = navis.prune_twigs(nl, size=10000)
    nl = navis.resample_skeleton(nl, resample_to=5000, inplace=False)

    objects = {
        "neurons": nl,
        "dps": navis.make_dotprops(nl, k=5),
        "vol": navis.simplify_mesh(navis.example_volume("LH"), F=0.05),
    }

    filepath = HERE / "objects.rda"
    navis.write_rda(objects, filepath)
    print(f"Wrote {filepath} ({filepath.stat().st_size} bytes)")


# Guarded: pytest runs with `--doctest-modules`, which imports every module it
# collects. Without this, writing the fixture would happen at import time - and
# fail on Python 3.10, where `rdata` cannot write.
if __name__ == "__main__":
    main()
