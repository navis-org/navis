"""
Code to run Jupyter notebooks in /docs/source
"""
import nbformat
import navis

from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor

SKIP = ['transforming.ipynb', 'python2cytoscape.ipynb', 'r_doc.ipynb',
        'neuprint.ipynb']

if __name__ == '__main__':
    navis.set_pbars(jupyter=False, hide=True)

    # Recursively search for notebooks
    path = Path(__file__).parent.parent / 'docs/source/tutorials'

    for file in path.glob('*.ipynb'):
        if not file.is_file():
            continue
        if file.name in SKIP:
            continue

        with open(file) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600)

        print('Executing', file.name)
        ep.preprocess(nb, {'metadata': {'path': file.parent}})

print('Done.')
