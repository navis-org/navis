from setuptools import setup, find_packages

import re

VERSIONFILE = "navis/__version__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if l and not l.startswith('#')]

LONG_DESCRIPTION = """
NAVis library is a Python 3 library for analysis and visualization of neuron
morphology.

Features include:

* work with various neuron types: skeletons, meshes, dotprops
* 2D (matplotlib) and 3D (vispy or plotly) plotting
* virtual neuron surgery (cutting, stitching, pruning, rerooting, ...)
* analyse morphology (e.g. NBLAST) and connectivity
* transform neurons between template brains
* Python bindings for R neuron libraries (e.g. nat, nat.nblast and elmr)
* load neurons directly from `neuromorpho.org <http://neuromorpho.org>`_ or :ref:`neuPrint<neuprint_intro>`
* interface with Blender 3D
* import-export from/to SWC
* extensible - see for example `pymaid <https://pymaid.readthedocs.io/en/latest/>`_

Check out the `Documentation <http://navis.readthedocs.io/>`_.
"""

setup(
    name='navis',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Neuron Analysis and Visualization Library',
    long_description=LONG_DESCRIPTION,
    url='http://navis.readthedocs.io',
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='Neuron Analysis Visualization Anatomy Connectivity',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=requirements,
    extras_require={'extras': []},
    python_requires='>=3.6',
    zip_safe=False,

    include_package_data=True

)
