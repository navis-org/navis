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

extras_require: DefaultDict[str, List[str]] = defaultdict(list)
install_requires: List[str] = []
reqs = install_requires

with open("requirements.txt") as f:
    for line in f:
        if line.startswith("#extra: "):
            extra = line[8:].split("#")[0].strip()
            reqs = extras_require[extra]
        elif not line.startswith("#") and line.strip():
            reqs.append(line.strip())

dev_only = ["test-notebook", "dev"]
extras_require["all"] = list(chain.from_iterable(
    v for k, v in extras_require.items() if k not in dev_only
))

Check out the `Documentation <http://navis.readthedocs.io/>`_.
"""

setup(
    name='navis',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Neuron Analysis and Visualization library',
    long_description=LONG_DESCRIPTION,
    url='http://navis.readthedocs.io',
    project_urls={
     "Documentation": "http://navis.readthedocs.io",
     "Source": "https://github.com/schlegelp/navis",
     "Changelog": "https://navis.readthedocs.io/en/latest/source/whats_new.html",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='Neuron Analysis Visualization Anatomy Connectivity Transform Neuroscience NBLAST Skeletons SWC neuPrint',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require),
    tests_require=extras_require["dev"],
    python_requires='>=3.7',
    zip_safe=False,

    include_package_data=True

)
