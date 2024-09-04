"""
Code to execute code examples notebooks in /docs/examples.

This will not be run through pytest but is meant to be run in a separate CI job.
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
            p = subprocess.run(["python", str(file)], check=True, capture_output=True, timeout=600)
        except subprocess.CalledProcessError as e:
            print("Failed.")
            print(e.stdout.decode())
            print(e.stderr.decode())
            raise
        print("Done.", flush=True)

    print("All done.")
