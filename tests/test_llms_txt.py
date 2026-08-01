"""Tests for `scripts/gen_llms_txt.py`, which builds the llms.txt family.

The generator parses `docs/api.md` rather than re-deriving the API grouping, so
the failure mode to guard against is silent: reformat a table in api.md and the
parser quietly yields nothing, leaving a technically-valid but empty llms.txt.
Most of what follows therefore pins the *shape* of the output - section count,
object count, size budget - rather than exact strings.
"""

import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_llms_txt.py"

# llms.txt is the tier that has to stay cheap enough to always fetch; the
# whole point of splitting the full docstrings out is that this file doesn't
# carry them. Generous headroom over the current ~55 kB so ordinary API growth
# doesn't trip it, tight enough to catch a regression that inlines docstrings
# or stops stripping type annotations from signatures.
INDEX_SIZE_BUDGET = 80_000
# The index must stay substantially smaller than the everything-blob, or the
# tiering has stopped buying anything.
MIN_FULL_TO_INDEX_RATIO = 5


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_llms_txt", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def files(gen):
    return gen.build()


def test_api_index_parses(gen):
    """api.md yields a plausible number of sections and objects."""
    sections = gen.parse_api_index()
    assert len(sections) >= 10, "api.md parsed into suspiciously few sections"

    n_objects = len({t for s in sections for t in s.targets})
    assert n_objects >= 350, f"only {n_objects} objects parsed out of api.md"

    titles = [s.title for s in sections]
    for expected in ("Neurons & NeuronLists", "Neuron Morphology", "Visualization"):
        assert expected in titles


def test_core_classes_are_indexed(gen):
    """The neuron classes are listed without the `autosummary` macro.

    They're the most important entries in the file and are easy to lose: their
    rows in api.md carry a plain cross-reference and a hand-written
    description, not the macro that every other row uses.
    """
    targets = {t for s in gen.parse_api_index() for t in s.targets}
    for cls in (
        "navis.TreeNeuron",
        "navis.MeshNeuron",
        "navis.VoxelNeuron",
        "navis.Dotprops",
        "navis.NeuronList",
    ):
        assert cls in targets


def test_expected_files_emitted(files):
    assert "llms.txt" in files
    assert "llms-full.txt" in files

    per_section = [n for n in files if n.startswith("llms-full-")]
    assert len(per_section) >= 8, "per-section files missing"
    assert "llms-full-neuron-morphology.txt" in per_section


def test_index_stays_small(files):
    index = files["llms.txt"]
    assert len(index) < INDEX_SIZE_BUDGET, (
        f"llms.txt is {len(index) / 1000:.0f} kB, over the "
        f"{INDEX_SIZE_BUDGET / 1000:.0f} kB budget - are docstrings or type "
        "annotations leaking into the index?"
    )
    assert len(files["llms-full.txt"]) > MIN_FULL_TO_INDEX_RATIO * len(index)


def test_index_covers_key_api(files):
    index = files["llms.txt"]
    for name in (
        "navis.nblast",
        "navis.make_dotprops",
        "navis.xform_brain",
        "navis.plot3d",
        "navis.prune_twigs",
    ):
        assert name in index, f"{name} missing from llms.txt"


def test_signatures_are_stripped_of_annotations(files):
    """Signatures carry parameter names and defaults, not type annotations."""
    index = files["llms.txt"]
    assert "navis.make_dotprops(x, k=20" in index
    # `self` is noise on methods reached through an instance.
    assert "navis.BaseNeuron.copy(deepcopy=False)" in index
    # A leaked annotation would drag `typing.` / `ForwardRef` in with it.
    assert "ForwardRef" not in index
    assert "typing.Union" not in index


def test_no_unresolved_entries(gen, files):
    """Every api.md target resolves, or is explained by a missing optional dep.

    `navis.interfaces.*` needs backends (neuron, py2cytoscape, caveclient, ...) that
    a plain test environment won't have; the docs CI installs them all. Those
    entries are expected to render as a labelled placeholder. Anything else
    failing to resolve means api.md points at something that doesn't exist.
    """
    for name, content in files.items():
        assert "(unresolved:" not in content, f"unresolved reference in {name}"

    # Whatever *did* fall back should be attributable to an absent submodule.
    for section in gen.parse_api_index():
        for target in section.targets:
            try:
                gen.resolve(target)
            except BaseException as exc:
                assert gen.unavailable_reason(target), (
                    f"api.md documents {target}, which does not resolve "
                    f"({type(exc).__name__}) and is not in an optional submodule"
                )


def test_core_api_always_resolves(gen, files):
    """No placeholders outside the interfaces/models namespaces."""
    for section in gen.parse_api_index():
        for target in section.targets:
            if target.startswith(("navis.interfaces.", "navis.models.")):
                continue
            gen.resolve(target)  # raises if api.md points at a dead name


def test_mkdocstrings_refs_are_flattened(files):
    """`[`navis.x`][]` is a link on the site and noise in a .txt."""
    for name, content in files.items():
        leftover = re.findall(r"\[[^\]]*\]\[[^\]]*\]", content)
        assert not leftover, f"unflattened cross-references in {name}: {leftover[:3]}"


def test_real_links_survive(files):
    """Flattening cross-references must not eat ordinary markdown links."""
    assert "](http" in files["llms-full.txt"]


def test_preamble_is_inlined(gen, files):
    """The hand-written idioms block reaches llms.txt.

    llms.txt is the only channel that reaches an agent working in someone
    else's project, so the guidance has to travel with the API index rather
    than sit in a repo file nobody fetches.
    """
    assert gen.PREAMBLE_MD.exists()

    index = files["llms.txt"]
    assert "## Read this first" in index
    for marker in ("Brain spaces & units", "NBLAST has three preconditions"):
        assert marker in index, f"preamble section {marker!r} missing"

    # Build-input comments are for maintainers, not for the wire.
    assert "<!--" not in index


def test_preamble_headings_are_demoted(gen, files):
    """`##` in the preamble must nest under llms.txt's `## Read this first`."""
    index = files["llms.txt"]
    assert "### Core model" in index
    assert "\n## Core model" not in index


def test_demotion_leaves_code_fences_alone(gen, files):
    """A line-initial `#` inside a code block is a comment, not a heading.

    Demoting it corrupts the example - and the xform_brain snippet, which is
    the single most important thing in the preamble, has one.
    """
    assert "# only now are hb and fw comparable" in files["llms.txt"]

    fenced = gen.demote_headings("## Real\n\n```python\n# a comment\nx = 1\n```\n")
    assert "### Real" in fenced
    assert "# a comment" in fenced and "## a comment" not in fenced


def test_type_compatibility_matrix_is_inlined(files):
    """The which-function-works-on-which-type table is copied verbatim.

    It answers a question agents routinely get wrong and exists nowhere else in
    machine-readable form.
    """
    index = files["llms.txt"]
    assert "Neuron types and functions" in index
    assert "| TreeNeuron | MeshNeuron | VoxelNeuron | Dotprops |" in index


def test_cli_writes_files(gen, tmp_path):
    gen.main(["--out", str(tmp_path)])
    written = {p.name for p in tmp_path.glob("*.txt")}
    assert "llms.txt" in written
    assert "llms-full.txt" in written
    assert (tmp_path / "llms.txt").read_text().startswith("# navis")


def test_cli_tolerates_foreign_argv(gen, tmp_path):
    """The gen-files plugin leaves mkdocs' own argv in place.

    Under `mkdocs build -q` this script sees `[<script>, 'build', '-q']`;
    argparse must ignore that rather than abort the docs build.
    """
    gen.main(["build", "-q", "--out", str(tmp_path)])
    assert (tmp_path / "llms.txt").exists()
