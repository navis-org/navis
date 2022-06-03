"""
Code to run Jupyter notebooks in /docs/source.
"""
import navis

from pathlib import Path

# If Jupyter is not installed, this will fail.
# We will belay the import error in case the
# user wants to run only the other parts of the
# test suite.
try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    nbformat = None
    pass
except BaseException:
    raise

SKIP = ['transforming.ipynb',  # requires navis-flybrains + transforms
        'python2cytoscape.ipynb',  # requires cytoscape
        'r_doc.ipynb',  # requires rpy2
        'nblast_flycircuit.ipynb',  # requires downloading flycircuit dotprops
        'nblast_hemibrain.ipynb',  # requires downloading data
        'microns_tut.ipynb',  # requires credentials
        'local_data_skeletons.ipynb',  # requires downloaded data
        'local_data_dotprops.ipynb',  # requires downloaded data
        'local_data_meshes.ipynb',  # requires downloaded data
        'local_data_voxels.ipynb',  # requires downloaded data
        'local_data_pickling.ipynb',  # requires downloaded data
        'neuromorpho_tut.ipynb',  # some certificate issue at the moment
        ]

if __name__ == '__main__':
    if not nbformat:
        raise ImportError('`nbformat` not found - please install Jupyter')

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

        print('Executing', file.name, '... ', end='')
        ep.preprocess(nb, {'metadata': {'path': file.parent}})
        print('Done.')

    print('All done.')
