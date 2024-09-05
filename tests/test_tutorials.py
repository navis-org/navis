"""
Code to execute code examples notebooks in /docs/examples.

This will not be run through pytest but is meant to be run in a separate CI job.

A couple notes:
 - it's possible that any notebook that spawns another child process (e.g. the SWC I/O tutorial)
   will hang indefinitely. This is because of the ominous "An attempt has been made to start a new
   process before the current process has finished its bootstrapping phase." error which typically
   means that the script has to be run in a `if __name__ == "__main__":` block.
   Set `capture_output=True` to see the error message.
 - the MICrONS tutorial occasionally fails because the CAVE backend throws an error
   (e.g. during the materialization)
"""

import subprocess
from pathlib import Path

SKIP = [

]

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
        try:
            # Set `capture_output=True` to see e.g. error messages.
            p = subprocess.run(["python", str(file)], check=True, capture_output=True, timeout=600, cwd=file.parent)
        except subprocess.CalledProcessError as e:
            print("Failed.")
            print(e.stdout.decode())
            print(e.stderr.decode())
            raise
        print("Done.", flush=True)

    print("All done.")
