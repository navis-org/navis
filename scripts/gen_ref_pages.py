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

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "navis"


#  Make the overview page
full_doc_path = "reference/overview.md"
with open('docs/api.md', 'r') as f:
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f.read())
nav[("overview",)] = 'overview.md'

# Parse modules
for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__version__":
        parts = parts[:-1]
    elif parts[-1] in ("__main__", "conftest"):
        continue

    if not len(parts):
        continue

    #print('!!!', parts, doc_path.as_posix())

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    print('  ', parts, doc_path, full_doc_path, path.relative_to(root))

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "a") as nav_file:
    #print(list(nav.build_literate_nav()))
    nav_file.writelines(nav.build_literate_nav())