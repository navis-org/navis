"""Generate the `llms.txt` family from the curated API index.

Agents (and the humans driving them) need the shape of the API in a single
cheap fetch. Three tiers get written to the docs root:

- `llms.txt`          the index: `llms_preamble.md` (the idioms an agent can't
                      infer from signatures) followed by every documented
                      object as `name(signature) - summary`. Small enough to
                      always afford (~15k tokens).
- `llms-full-<sec>.txt`  one file per top-level API section, with complete
                      docstrings. Fetched selectively - an agent that wants
                      NBLAST detail shouldn't pay for the plotting docstrings.
- `llms-full.txt`     everything, for crawlers that want one blob. Too large
                      to read whole (~100k+ tokens); the per-section files are
                      the intended path.

The grouping is *not* re-derived here: it is parsed out of `docs/api.md`, which
already curates the whole surface into task-shaped sections via the
`autosummary("navis.x.y")` macro (see `main.py`). That keeps one source of
truth - a function added to `api.md` shows up here automatically, and one that
never made it into `api.md` gets reported as uncovered rather than silently
dropped.

Runs two ways:

- under mkdocs, via the `gen-files` plugin (writes into the built site);
- standalone, `python scripts/gen_llms_txt.py --out DIR`, which needs nothing
  but navis and is what the tests use.
"""

import argparse
import importlib
import inspect
import re
import sys
import warnings
from pathlib import Path

import navis

# `[`navis.foo`][]`, `[`foo()`][navis.foo]`, `[navis.foo][]` - mkdocstrings'
# cross-reference syntax. Deliberately does not match `[text](url)`.
MKDOCSTRINGS_REF_RE = re.compile(r"\[`?([^\]`]+?)`?\]\[([^\]]*)\]")

ROOT = Path(__file__).parent.parent
API_MD = ROOT / "docs" / "api.md"
# The hand-written half of llms.txt: the idioms that can't be generated from
# docstrings. Lives next to this script because it is a build input, not a page.
PREAMBLE_MD = Path(__file__).parent / "llms_preamble.md"

SITE_URL = "https://navis-org.github.io/navis"
REPO_URL = "https://github.com/navis-org/navis"

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

# `## Heading` in api.md that is prose/tables rather than a list of objects,
# but is worth copying into llms.txt verbatim - the type-compatibility matrix
# is exactly the "will this function work on this neuron?" question agents get
# wrong.
VERBATIM_SECTIONS = ("Neuron types and functions",)

# Sub-namespaces that `import navis` does not pull in but that api.md
# documents. Imported defensively at build time: a missing optional dependency
# should cost us that module's entries, not the whole file.
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


# --------------------------------------------------------------------------- #
# Resolving + rendering individual objects
# --------------------------------------------------------------------------- #


#: Optional submodules that could not be imported in *this* environment.
#: Populated by `import_optional_submodules`; used to explain unresolvable
#: entries as a missing dependency rather than a broken reference.
_MISSING_SUBMODULES: set[str] = set()


def import_optional_submodules() -> set[str]:
    """Import the sub-namespaces `import navis` does not pull in.

    api.md documents them, so without this every `navis.interfaces.*` entry is
    unresolvable. The docs CI installs all the optional dependencies and so
    resolves everything; a partial environment gets a labelled placeholder for
    whatever is missing instead of a failed build.
    """
    _MISSING_SUBMODULES.clear()
    for name in OPTIONAL_SUBMODULES:
        try:
            importlib.import_module(name)
        except BaseException:
            # Deliberately broad: these raise ImportError subclasses,
            # RuntimeError, or even SystemExit (the Blender API outside of
            # Blender) when their backend is unavailable.
            _MISSING_SUBMODULES.add(name)
    return _MISSING_SUBMODULES


def unavailable_reason(dotted: str) -> str | None:
    """Which missing optional submodule, if any, explains `dotted`."""
    for missing in _MISSING_SUBMODULES:
        if dotted == missing or dotted.startswith(missing + "."):
            return missing
    return None


def resolve(dotted: str):
    """Resolve a dotted path such as `navis.morpho.prune_twigs` to its object.

    Mirrors the `autosummary` macro in `main.py` so that anything renderable in
    `api.md` is renderable here.
    """
    obj = navis
    for part in dotted.split("."):
        if part == "navis":
            continue
        obj = getattr(obj, part)
    return obj


def signature(obj) -> str:
    """Compact signature: parameter names and defaults, no annotations.

    Annotations are stripped deliberately. `plot3d`'s is a 500-character Union
    that costs real context and tells an agent less than the parameter names
    do; the full annotated signature is one fetch away in the per-section file.
    """
    if isinstance(obj, property) or not callable(obj):
        return ""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"

    params = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    # Methods are reached through an instance (`neuron.copy()`), so the bound
    # first argument is noise at best and misleading at worst.
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    try:
        return str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))
    except (TypeError, ValueError):
        return "(...)"


def clean_refs(text: str) -> str:
    """Flatten mkdocstrings cross-references to plain dotted names.

    ``[`navis.plot3d`][]`` and ``[`plot3d()`][navis.plot3d]`` both render as a
    link on the docs site but are noise in a .txt. Ordinary markdown links
    (`[text](url)`) carry information an agent can act on and are left alone.
    """
    return MKDOCSTRINGS_REF_RE.sub(
        lambda m: (m.group(2) or m.group(1)).removesuffix("()"), text
    )


def summary(obj) -> str:
    """First non-empty line of the docstring."""
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        if line.strip():
            return clean_refs(line.strip())
    return ""


def docstring(obj) -> str:
    return clean_refs(inspect.getdoc(obj) or "")


def kind(obj) -> str:
    if isinstance(obj, property):
        return "property"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "attribute"


# --------------------------------------------------------------------------- #
# Parsing the curated index out of docs/api.md
# --------------------------------------------------------------------------- #

# api.md lists objects exclusively in markdown tables - one row per object, the
# name in the first cell. Two row dialects exist and both are load-bearing:
#
#   | [`navis.prune_twigs()`][navis.prune_twigs] | {{ autosummary("navis.prune_twigs") }} |
#   | [`navis.TreeNeuron`][]                     | Skeleton representation of a neuron.   |
#
# Keying only off the `autosummary` macro misses the second dialect, which is
# how the core neuron classes are listed. Restricting to table rows (rather
# than scanning all cross-references) keeps incidental "see also" links in
# prose from being pulled in as members of whatever section they sit under.
AUTOSUMMARY_RE = re.compile(r'autosummary\(\s*["\']([\w\.]+)["\']\s*\)')
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
XREF_RE = re.compile(r"\[`([^`]+)`\]\[([^\]]*)\]")
TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|?$")


class Section:
    """One `## `-level chunk of api.md, with its `### ` subsections."""

    def __init__(self, title: str):
        self.title = title
        self.groups: list[tuple[str, list[str]]] = []  # (subtitle, [dotted paths])
        self.verbatim: list[str] = []

    @property
    def slug(self) -> str:
        s = re.sub(r"[^\w\s-]", "", self.title.lower())
        return re.sub(r"[\s_]+", "-", s).strip("-")

    @property
    def targets(self) -> list[str]:
        return [t for _, members in self.groups for t in members]

    def __repr__(self):
        return f"<Section {self.title!r} n={len(self.targets)}>"


def row_targets(line: str) -> list[str]:
    """Dotted paths documented by one markdown table row, in order.

    Cross-references are only read out of the *first* cell so that a
    description mentioning another function doesn't enrol it into this section.
    """
    stripped = line.strip()
    if not stripped.startswith("|") or TABLE_SEP_RE.match(stripped):
        return []

    out: list[str] = []
    first_cell = stripped.strip("|").split("|")[0]
    for display, explicit in XREF_RE.findall(first_cell):
        target = (explicit or display).removesuffix("()")
        if target.startswith("navis."):
            out.append(target)
    for target in AUTOSUMMARY_RE.findall(stripped):
        if target not in out:
            out.append(target)
    return out


def parse_api_index(path: Path = API_MD) -> list[Section]:
    """Parse api.md into ordered sections of dotted object paths.

    Duplicates are dropped per section (a name listed in two tables of the same
    section) but kept across sections - `navis.plot3d` legitimately belongs to
    both "Visualization" and a neuron-type overview.
    """
    sections: list[Section] = []
    current: Section | None = None
    subtitle = ""
    seen_in_section: set[str] = set()
    collecting_verbatim = False

    for line in path.read_text().splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            if level == 2:
                current = Section(title)
                sections.append(current)
                subtitle = ""
                seen_in_section = set()
                collecting_verbatim = title in VERBATIM_SECTIONS
            elif current is not None:
                subtitle = title
            continue

        if current is None:
            continue

        if collecting_verbatim:
            current.verbatim.append(XREF_RE.sub(lambda m: m.group(2) or m.group(1), line))
            continue

        found = row_targets(line)
        if not found:
            continue
        for target in found:
            if target in seen_in_section:
                continue
            seen_in_section.add(target)
            if current.groups and current.groups[-1][0] == subtitle:
                current.groups[-1][1].append(target)
            else:
                current.groups.append((subtitle, [target]))

    return [s for s in sections if s.targets or any(l.strip() for l in s.verbatim)]


# --------------------------------------------------------------------------- #
# Coverage: public objects api.md never mentions
# --------------------------------------------------------------------------- #


def public_top_level() -> set[str]:
    """Public objects reachable as `navis.<name>`."""
    out = set()
    for name in dir(navis):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(navis, name)
        except BaseException:
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj) or _is_wrapped(obj):
            if getattr(obj, "__module__", "").startswith("navis"):
                out.add(f"navis.{name}")
    return out


def _is_wrapped(obj) -> bool:
    """`lru_cache`-style wrappers are callable but not `inspect.isfunction`."""
    try:
        return callable(obj) and hasattr(obj, "__wrapped__")
    except BaseException:
        return False


def uncovered(sections: list[Section]) -> list[str]:
    documented = {t for s in sections for t in s.targets}
    return sorted(public_top_level() - documented)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def preamble() -> str:
    """The hand-written idioms block that opens llms.txt."""
    if not PREAMBLE_MD.exists():
        warnings.warn(
            f"{PREAMBLE_MD.name} is missing; llms.txt will ship as a bare API "
            "index, without the idioms an agent can't infer from signatures."
        )
        return ""
    body = HTML_COMMENT_RE.sub("", PREAMBLE_MD.read_text())
    return demote_headings(body.strip())


def demote_headings(text: str) -> str:
    """Push every markdown heading down one level, skipping code fences.

    The preamble nests under llms.txt's `## Read this first`, so its `##`
    become `###`. Fenced blocks have to be skipped or a line-initial Python
    comment (`# only now are hb and fw comparable`) gets rewritten into a
    heading, silently corrupting the example.
    """
    out, in_fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"^#+ ", line):
            line = "#" + line
        out.append(line)
    return "\n".join(out)


def render_index(sections: list[Section], missing: list[str]) -> str:
    out: list[str] = []
    w = out.append

    w("# navis")
    w("")
    w("> Python library for the analysis of neuroanatomical data: skeletons, meshes,")
    w("> voxel images and dotprops, with morphometrics, NBLAST, template-brain")
    w("> transforms, connectivity analysis and 2D/3D plotting.")
    w("")
    w(f"Version: {navis.__version__}")
    w(f"Docs: {SITE_URL}/  |  Repo: {REPO_URL}")
    w("")
    w("This is the condensed API index: every documented object with its parameter")
    w("names and one-line summary. Signatures are shown without type annotations.")
    w("For complete docstrings fetch the per-section file named under each heading")
    w(f"(e.g. `{SITE_URL}/llms-full-neuron-morphology.txt`). `llms-full.txt` holds")
    w("all of them concatenated and is large - prefer the per-section files.")
    w("")

    idioms = preamble()
    if idioms:
        w("## Read this first")
        w("")
        w(idioms)
        w("")

    w("## API index")
    w("")

    for section in sections:
        w(f"### {section.title}")
        if section.targets:
            w("")
            w(f"Full docstrings: {SITE_URL}/llms-full-{section.slug}.txt")
        w("")
        if section.verbatim:
            w("\n".join(section.verbatim).strip())
            w("")
            continue
        for subtitle, members in section.groups:
            if subtitle:
                w(f"#### {subtitle}")
                w("")
            for target in members:
                w(render_entry(target))
            w("")

    if missing:
        w("### Not covered above")
        w("")
        w("Public objects that are not yet in the curated index. They exist and work;")
        w("call `help(navis.<name>)` for details.")
        w("")
        w(", ".join(f"`{m}`" for m in missing))
        w("")

    return "\n".join(out).rstrip() + "\n"


def render_entry(target: str) -> str:
    try:
        obj = resolve(target)
    except BaseException as exc:
        missing = unavailable_reason(target)
        if missing:
            return f"- `{target}` - requires the optional `{missing}` dependencies"
        return f"- `{target}` - (unresolved: {type(exc).__name__})"
    line = summary(obj)
    if isinstance(obj, property) or not callable(obj):
        return f"- `{target}` - {line}" if line else f"- `{target}`"
    return f"- `{target}{signature(obj)}` - {line}"


def render_full(section: Section) -> str:
    out: list[str] = []
    w = out.append

    w(f"# navis {navis.__version__} - {section.title}")
    w("")
    w(f"Full docstrings for the '{section.title}' section of the navis API.")
    w(f"Index of the whole API: {SITE_URL}/llms.txt")
    w("")

    if section.verbatim:
        w("\n".join(section.verbatim).strip())
        w("")

    for subtitle, members in section.groups:
        if subtitle:
            w(f"## {subtitle}")
            w("")
        for target in members:
            try:
                obj = resolve(target)
            except BaseException as exc:
                missing = unavailable_reason(target)
                w(f"### {target}")
                w("")
                w(
                    f"Requires the optional `{missing}` dependencies, which were "
                    "not installed when these docs were built."
                    if missing
                    else f"(unresolved: {type(exc).__name__}: {exc})"
                )
                w("")
                continue
            w(f"### {target}{signature(obj)}")
            w("")
            w(f"*{kind(obj)}*")
            w("")
            doc = docstring(obj)
            if doc:
                w(doc)
                w("")

    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build() -> dict[str, str]:
    """Build every llms*.txt file. Returns {filename: content}."""
    import_optional_submodules()
    sections = parse_api_index()
    missing = uncovered(sections)

    files = {"llms.txt": render_index(sections, missing)}

    full_parts = []
    for section in sections:
        if not section.targets:
            continue
        content = render_full(section)
        files[f"llms-full-{section.slug}.txt"] = content
        full_parts.append(content)

    files["llms-full.txt"] = (
        f"# navis {navis.__version__} - full API reference\n\n"
        f"Every documented object with its complete docstring. This file is large;\n"
        f"the per-section `llms-full-<section>.txt` files hold the same content split\n"
        f"by topic, and `{SITE_URL}/llms.txt` is the condensed index.\n\n"
        + "\n\n".join(full_parts)
    )
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write files to this directory")
    parser.add_argument(
        "--stats", action="store_true", help="print size and coverage stats to stderr"
    )
    # The gen-files plugin runs this through `runpy` without resetting
    # `sys.argv`, so under `mkdocs build -q` argv is
    # `[<this script>, 'build', '-q']`. Ignore what we don't recognise rather
    # than letting argparse abort the docs build over mkdocs' own flags.
    args, _ = parser.parse_known_args(argv)

    files = build()

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (args.out / name).write_text(content)
    else:
        try:
            import mkdocs_gen_files
        except ImportError:
            parser.error(
                "no --out given and mkdocs_gen_files is unavailable; pass "
                "--out DIR to run outside of a docs build"
            )
        for name, content in files.items():
            with mkdocs_gen_files.open(name, "w") as fh:
                fh.write(content)

    if args.stats:
        sections = parse_api_index()
        missing = uncovered(sections)
        print(f"{'file':<44} {'kB':>8} {'~tokens':>9}", file=sys.stderr)
        for name, content in sorted(files.items()):
            print(
                f"{name:<44} {len(content) / 1000:>8.1f} {len(content) / 4 / 1000:>8.1f}k",
                file=sys.stderr,
            )
        n_documented = len({t for s in sections for t in s.targets})
        print(
            f"\n{len(sections)} sections, {n_documented} documented objects, "
            f"{len(missing)} public top-level objects uncovered",
            file=sys.stderr,
        )
        if missing:
            print(f"uncovered: {', '.join(missing)}", file=sys.stderr)

    return files


# `runpy` names the gen-files execution `<run_path>`; a direct run is
# `__main__`. Both should build. A plain `import gen_llms_txt` - which is what
# the tests do - must not.
if __name__ in ("__main__", "<run_path>"):
    main()
