"""
Code to execute code tutorial notebooks in /docs/examples/.

This will not be run through pytest but is executed in a separate CI job.

A couple notes:
 - the tutorials require a number of extra dependencies and data files to be present
   check out the test-tutorials.yml workflow to see how this is set up.
 - the MICrONS tutorial occasionally fails because the CAVE backend throws an error
   (e.g. during the materialization)
 - Github runners appear to have 4 CPUs - so should be good to go
"""

import os
import sys
import navis
import warnings

import matplotlib.pyplot as plt

from pathlib import Path
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

TO_SKIP = [
    "tutorial_remote_04_h01.py",  # currently fails due to incompatability with most recent CAVEclient
]

# Tutorials living under this subdirectory pull data from third-party services
# (neuPrint, CAVE/MICrONS, InsectBrainDB, BIL, ...) and are therefore prone to
# transient failures that are outside our control.
REMOTE_SUBDIR = "4_remote"


def _collect_files(args):
    """Collect the tutorial files to execute.

    Positional arguments (if any) are treated as paths - either directories
    (searched recursively for `*.py`) or individual files - and let the CI
    split the deterministic tutorials from the flaky remote ones into separate
    jobs. Without arguments we fall back to running everything in
    `docs/examples`.

    Set `NAVIS_SKIP_REMOTE=1` to skip everything under the `4_remote`
    subdirectory (used by the required, non-remote CI job).
    """
    default_path = Path(__file__).parent.parent / "docs/examples"

    if args:
        files = []
        for arg in args:
            p = Path(arg)
            if not p.is_absolute():
                # Resolve relative to the cwd first, then the repo root.
                p = p if p.exists() else default_path.parent.parent / arg
            if p.is_dir():
                files.extend(sorted(p.rglob("*.py")))
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f"No such tutorial path: {arg}")
    else:
        files = list(default_path.rglob("*.py"))

    if os.environ.get("NAVIS_SKIP_REMOTE"):
        files = [f for f in files if REMOTE_SUBDIR not in f.parts]

    return files


if __name__ == "__main__":
    # N.B. these are deliberately *not* at module level. This file matches
    # `test_*.py`, so the main pytest job imports it even though it holds no
    # tests - and silencing the logger/warnings there would leak into every
    # other test in the session (it did: it broke tests asserting on warnings).
    navis.config.logger.setLevel("ERROR")
    navis.set_pbars(hide=True)
    warnings.filterwarnings("ignore")

    files = _collect_files(sys.argv[1:])
    for i, file in enumerate(files):
        if not file.is_file():
            continue
        if file.name.startswith('zzz'):
            continue
        if file.name in TO_SKIP:
            print(f"Skipping {file.name}...")
            continue

        # Note: we're using `exec` here instead of e.g. `subprocess.run` because we need to avoid
        # the "An attempt has been made to start a new process before the current process has
        # finished its bootstrapping phase" error that occurs when using multiprocessing with "spawn"
        # from a process where it's not wrapped in an `if __name__ == "__main__":` block.
        print(f"Executing {file.name} [{i+1}/{len(files)}]... ", end="", flush=True)
        with suppress_stdout():
            os.chdir(file.parent)
            exec(open(file.name).read())
        print("Done.", flush=True)

        # Make sure to close any open figures
        plt.close("all")

    print("All done.")
