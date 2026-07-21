"""
This script crawls the package space and generates .md files in
the references directory of the mkdocs site. For example:

```
# side/reference/connectivity/adjacency.md
::: connectivity.adjacency
```

It also generates a SUMMARY.MD file that contains the
navigation structure of the generated .md files. We
then use this SUMMARY.md file in the mkdocs.yml file to
add the generated pages to the site.

Modified from https://mkdocstrings.github.io/recipes/#generate-a-literate-navigation-file

"""
import importlib
import warnings

import navis

# Submodules that aren't imported by default. These pull in optional
# third-party dependencies (neuron, caveclient, rpy2, ...) and some of them
# raise at import time when those are missing. Import them defensively: a
# missing optional dependency should cost us that module's API reference page,
# not the entire docs build.
OPTIONAL_SUBMODULES = (
    "navis.models",
    "navis.interfaces.neuron",
    "navis.interfaces.neuromorpho",
    "navis.interfaces.neuprint",
    "navis.interfaces.insectbrain_db",
    "navis.interfaces.blender",
    "navis.interfaces.microns",
    "navis.interfaces.h01",
    "navis.interfaces.brain_image_library",
    "navis.interfaces.cytoscape",
    "navis.interfaces.r",
)

for _module in OPTIONAL_SUBMODULES:
    try:
        importlib.import_module(_module)
    except BaseException as _exc:
        # Deliberately broad: these modules like to raise ImportError
        # subclasses, RuntimeError or even SystemExit when their backend is
        # unavailable (e.g. the Blender API outside of Blender).
        warnings.warn(
            f"Could not import {_module} ({type(_exc).__name__}: {_exc}). "
            "Its API reference page will be missing from the docs."
        )


import inspect
from pathlib import Path

import mkdocs_gen_files


# We will now traverse the navis package namespace in a
# breadth first search manner and collect all functions
# and classes where they show up first in the tree.
funcs = {}  # track functions and classes we've found so far
seen = []

# Set to True to print debug information
verbose = False

print_org = print  # Save the original print function

def print(*args, **kwargs):
    if verbose:
        print_org(*args, **kwargs)


def _safe_hasattr(obj, name):
    """`hasattr` that also survives objects raising non-AttributeError."""
    try:
        return hasattr(obj, name)
    except BaseException:
        return False


def _is_wrapped(obj):
    """Is this a decorated function that is no longer a plain function?

    `functools.lru_cache` (and friends) return an object that is callable and
    carries the original function's `__name__`/`__module__` but does *not*
    satisfy `inspect.isfunction`. Without this, every `@lru_cache`-decorated
    public function silently vanishes from the API reference.
    """
    return callable(obj) and _safe_hasattr(obj, '__wrapped__')

# Here we collect a dictionary of {function: module}
to_traverse = [navis]
root = Path(__file__).parent.parent
print("Starting at Root", root)
while to_traverse:
    current = to_traverse.pop(0)
    print("Traversing", current)
    # if current.__name__ == "navis.transforms.templates":
    #     printv('!!!', dir(current))

    for name in dir(current):
        print("  Checking", name, '... ' , end='')
        # Skip private functions, classes, etc
        if name[0] == '_':
            print('skipped (private).')
            continue
        try:
            obj = getattr(current, name)
        except BaseException as exc:
            # Some libraries (e.g. trimesh) park a placeholder object under the
            # name of an optional dependency that re-raises the original import
            # error on *any* attribute access. Don't let that kill the crawl.
            print(f'skipped (raised {type(exc).__name__} on access).')
            continue

        if inspect.ismodule(obj):
            print('module...', end='')
            #Skip things that aren't part of navis
            try:
                if hasattr(obj, '__path__') and not obj.__path__[0].startswith(str(root)):
                    continue
                elif hasattr(obj, '__file__') and not obj.__file__.startswith(str(root)):
                    continue
            except BaseException:
                print('Error parsing', obj)
                continue

            if obj not in seen:
                print('marked for traversal')
                seen.append(obj)
                to_traverse.append(obj)
            else:
                print('skipped (already seen).')
        # In some cases the object's name may not actually be defined in the module
        # E.g. named tuples
        elif not _safe_hasattr(obj, '__name__') or not _safe_hasattr(current, obj.__name__):
            print('skipped (not defined).')
            continue
        elif inspect.isfunction(obj) or inspect.isclass(obj) or _is_wrapped(obj):
            # Skip things that aren't part of navis
            if not getattr(obj, '__module__', '').startswith('navis'):
                print('skipped (not navis).')
                continue

            # Check if that function has been seen before
            if obj in funcs:
                print('skipped (already seen).')
                continue

            # If this is the first time we see the function,
            # add it to the list and track where it was found
            funcs[obj] = current
            print('added to list.')
        else:
            print('skipped (not function, class, or module).')

# Collate functions by module
by_module = {}
for k, v in funcs.items():
    by_module[v.__name__] = by_module.get(v.__name__, []) + [k.__name__]

# Sort modules by depth such that e.g. `navis` shows
# up first, then `navis.plotting`, etc.
# Here, we can also enforce e.g. a maximum depth
by_module = {k: v for k, v in sorted(by_module.items(), key=lambda x: len(x[0].split('.')))}


nav = mkdocs_gen_files.Nav()
src = root / "navis"
# Now we need to write the .md files
for module, functions in by_module.items():
    parts = tuple(module.split('.'))
    doc_path = Path('/'.join(parts) + ".md")
    full_doc_path = "reference/" / doc_path
    path = src / ('/'.join(parts) + ".py")

    #print('!!!', parts, doc_path.as_posix())

    print(f"Documenting {module}: {functions} in {full_doc_path}")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        for f in functions:
            print(f"::: {module}.{f}\n")
            print_org(f"::: {module}.{f}\n", file=fd)

    # print('  ', parts, doc_path, full_doc_path, path.relative_to(root))

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "a") as nav_file:
    print('\n\n\n', ','.join(list(nav.build_literate_nav())))
    nav_file.writelines(nav.build_literate_nav())