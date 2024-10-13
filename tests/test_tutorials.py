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

# Silence logging
navis.config.logger.setLevel("ERROR")

# Hide pbars
navis.set_pbars(hide=True)

# Silence warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Recursively search for notebooks
    path = Path(__file__).parent.parent / "docs/examples"

    files = list(path.rglob("*.py"))
    for i, file in enumerate(files):
        if not file.is_file():
            continue
        if file.name.startswith('zzz'):
            continue

        # Note: we're using `exec` here instead of e.g. `subprcoess.run` because we need to avoid
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
