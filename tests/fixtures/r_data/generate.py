"""Regenerate `objects.rda`.

Run with `python tests/fixtures/r_data/generate.py` from the repo root. Needs
`rdata` >= 1.0 (i.e. Python >= 3.11) since that is what can write R data files.

The fixture exists so that the *reading* side of `navis.io.rda_io` is covered
on Python 3.10, where `rdata` is too old to write and every roundtrip test is
skipped. Neurons are pruned and resampled to keep the file small while still
carrying nodes, connectors, dotprops and a mesh.
"""

from pathlib import Path

import navis

navis.set_pbars(hide=True)

HERE = Path(__file__).resolve().parent

nl = navis.example_neurons(2, kind="skeleton")
nl = navis.prune_twigs(nl, size=10000)
nl = navis.resample_skeleton(nl, resample_to=5000, inplace=False)

objects = {
    "neurons": nl,
    "dps": navis.make_dotprops(nl, k=5),
    "vol": navis.simplify_mesh(navis.example_volume("LH"), F=0.05),
}

navis.write_rda(objects, HERE / "objects.rda")
print(f"Wrote {HERE / 'objects.rda'} ({(HERE / 'objects.rda').stat().st_size} bytes)")
