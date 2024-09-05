"""
Code to execute code tutorial notebooks in /docs/examples/.

This will not be run through pytest but is executed in a separate CI job.

A couple notes:
 - the tutorials require a number of extra dependencies and data files to be present
   check out the test-tutorials.yml workflow to see how this is set up.
 - it's possible that any notebook that spawns another child process (e.g. the SWC I/O tutorial)
   will hang indefinitely. This is because of the ominous "An attempt has been made to start a new
   process before the current process has finished its bootstrapping phase." error which typically
   means that the script has to be run in a `if __name__ == "__main__":` block.
   Set `capture_output=True` to see the error message.
 - the MICrONS tutorial occasionally fails because the CAVE backend throws an error
   (e.g. during the materialization)
 - Github runners appear to have 4 CPUs - so should be good to go
"""

import os
import sys
import navis

from pathlib import Path
from contextlib import contextmanager

SKIP = ["zzz_no_plot_01_nblast_flycircuit.py", "zzz_no_plot_02_nblast_hemibrain.py"]


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
navis.set_pbars(hide=True)


if __name__ == "__main__":
    # Recursively search for notebooks
    path = Path(__file__).parent.parent / "docs/examples"

    files = list(path.rglob("*.py"))
    for i, file in enumerate(files):
        if not file.is_file():
            continue
        if file.name in SKIP:
            continue

        print(f"Executing {file.name} [{i+1}/{len(files)}]... ", end="", flush=True)
        with suppress_stdout():
            exec(open(file).read())
        print("Done.", flush=True)

    print("All done.")
