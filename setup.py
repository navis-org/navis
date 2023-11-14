from setuptools import setup, find_packages
from pathlib import Path
from runpy import run_path

from extreqs import parse_requirement_files

HERE = Path(__file__).resolve().parent

verstr = run_path(str(HERE / "navis" / "__version__.py"))["__version__"]

install_requires, extras_require = parse_requirement_files(
    HERE / "requirements.txt",
)

dev_only = ["test-notebook", "dev"]
specialized = ['r']
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

vispy_req = extras_require["vispy-default"][0]
default_backend = vispy_req.split("[")[1].split("]")[0]
# listed here: https://vispy.org/installation.html#backend-requirements
for alt in [
    "pyglet",
    "pyqt5", "pyqt6",
    "pyside", "pyside2", "pyside6",
    "glfw", "sdl2", "wx", "tk",
]:
    extras_require[f"vispy-{alt}"] = [vispy_req.replace(default_backend, alt)]

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
    url='http://navis.readthedocs.io',
    project_urls={
     "Documentation": "http://navis.readthedocs.io",
     "Source": "https://github.com/navis-org/navis",
     "Changelog": "https://navis.readthedocs.io/en/latest/source/whats_new.html",
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        # 'Programming Language :: Python :: 3.12',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require),
    tests_require=extras_require["dev"],
    # CI runs against >=3.8
    python_requires='>=3.9,<4.0',
    zip_safe=False,

    include_package_data=True

)
