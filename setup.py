from setuptools import setup, find_packages
from pathlib import Path
from runpy import run_path

from extreqs import parse_requirement_files

HERE = Path(__file__).resolve().parent

verstr = run_path(str(HERE / "navis" / "__version__.py"))["__version__"]

install_requires, extras_require = parse_requirement_files(
    HERE / "requirements.txt",
)

dev_only = ["test-notebook", "dev", "docs"]
# Kept out of `[all]`: needed only to point navis at a cluster, and dask pulls
# in a sizeable dependency tree that a local install has no use for.
# N.B. these are *extra* names as declared by `#extra:` in requirements.txt,
# not distribution names - the cloud-volume package sits under `cloudvolume`.
specialized = ["flybrains", "cloudvolume", "cluster"]
all_dev_deps = []
all_deps = []
for k, v in extras_require.items():
    if k in specialized:
        continue
    all_dev_deps.extend(v)
    if k not in dev_only:
        all_deps.extend(v)

extras_require["all"] = all_deps
extras_require["all-dev"] = all_dev_deps

with open("README.md") as f:
    long_description = f.read()

setup(
    name='navis',
    version=verstr,
    packages=find_packages(include=["navis", "navis.*"]),
    license='GNU GPL V3',
    description='Neuron Analysis and Visualization library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://navis-org.github.io/navis/',
    project_urls={
     "Documentation": "https://navis-org.github.io/navis/",
     "Source": "https://github.com/navis-org/navis",
     "Changelog": "https://navis-org.github.io/navis/changelog/",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='Neuron Analysis Visualization Morphometrics Morphology Anatomy Connectivity Transform Neuroscience NBLAST Skeletons SWC neuPrint',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require),
    tests_require=extras_require["dev"],
    # CI runs against >=3.10
    python_requires='>=3.10,<4.0',
    zip_safe=False,

    include_package_data=True

)
