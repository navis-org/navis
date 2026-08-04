"""
Tier-2 prototype: build the tutorial gallery index ourselves.

`mkdocs-gallery` still does all the heavy lifting - it executes every tutorial,
renders the per-tutorial pages, exports the ``.py`` / ``.ipynb`` downloads and
generates the thumbnails. This hook replaces *only* the rendered content of the
gallery landing page (``generated/gallery/index.md``) with a custom layout we
fully control.

How it plugs in
---------------
Registered as a native mkdocs hook (see ``hooks:`` in ``mkdocs.yml``). Hooks run
*after* every plugin for a given event, so by the time our ``on_page_markdown``
fires the gallery plugin has already produced the thumbnails on disk and the
macros plugin has already run - which is why we expand ``{{ navis }}`` ourselves
here instead of leaving it to the macro engine.

We rebuild the card metadata straight from the sources:
  * title   -> first line of the tutorial docstring
  * blurb   -> first paragraph of the docstring
  * level   -> ``<!-- difficulty: beginner -->`` in the docstring (see below)
  * thumb   -> ``<section>/images/thumb/mkd_glr_<stem>_thumb.png`` (the plugin
               normalises every thumbnail - scraped, pinned or default - to this
               path, so we never have to parse ``_thumbnail_*`` directives)
  * url     -> ``<section>/<stem>`` (directory-URL, relative to the index page)
  * section -> the sub-folder + its ``README.md`` heading/intro

Difficulty tags
---------------
Every tutorial declares its own level in its header, directly under the title::

    Manipulate Morphology
    =====================
    <!-- difficulty: beginner -->

    Prune, resample, smooth and reshape neuron morphology.

One of ``beginner`` / ``intermediate`` / ``advanced``; a missing or unknown
value logs a warning and simply renders no chip. An HTML comment is used so the
marker is invisible wherever the docstring is rendered (the tutorial page, the
exported notebook) while still sitting in the header where an author will find
it. It goes *under* the ``===`` underline so it cannot end up in the paragraph
`mkdocs_gallery` uses as the example's introduction.

Prose (the landing header and each section heading/intro) stays sourced from the
``README.md`` files, so editing copy still happens there, not in this script.

To disable / fall back to the stock mkdocs-gallery index: remove this file from
the ``hooks:`` list in ``mkdocs.yml``.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.gallery_index")

# The page whose content we take over (source path relative to docs_dir).
GALLERY_INDEX = "generated/gallery/index.md"

# Only these files are gallery examples (mirrors `filename_pattern: "/tutorial_"`).
TUTORIAL_GLOB = "*tutorial_*.py"


# --------------------------------------------------------------------------- #
# Self-contained styles for the prototype. Kept inline so the whole experiment
# lives in one file and is trivial to remove. Uses Material's CSS variables so
# it themes correctly in both light and dark mode.
# --------------------------------------------------------------------------- #
STYLE = """
<style>
.gallery-jump {
  display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.2rem 0 2rem;
}
.gallery-jump a {
  font-size: .78rem; padding: .25rem .75rem; border-radius: 1rem;
  background: var(--md-default-fg-color--lightest);
  color: var(--md-default-fg-color); text-decoration: none;
}
.gallery-jump a:hover {
  background: var(--md-accent-fg-color); color: var(--md-accent-bg-color);
}
.gallery-grid {
  display: grid; gap: 1rem; margin: 1rem 0 2.5rem;
  grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
}
.gallery-card {
  display: flex; flex-direction: column;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: .55rem; overflow: hidden;
  background: var(--md-default-bg-color);
  color: inherit; text-decoration: none;
  transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
}
.gallery-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 6px 18px rgba(0, 0, 0, .16);
  border-color: var(--md-accent-fg-color);
}
.gallery-card img {
  width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block;
  background: var(--md-default-fg-color--lightest);
}
.gallery-card-body {
  display: flex; flex-direction: column; gap: .3rem; padding: .65rem .75rem .8rem;
}
.gallery-card-title { font-weight: 600; font-size: .9rem; line-height: 1.25; }
.gallery-card-desc {
  font-size: .72rem; line-height: 1.35; opacity: .72;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  overflow: hidden;
}
.gallery-badges { display: flex; flex-wrap: wrap; gap: .3rem; }
.gallery-badge {
  font-size: .54rem; text-transform: uppercase; letter-spacing: .035em;
  padding: .1rem .38rem; border-radius: .25rem;
  background: var(--md-default-fg-color--lightest);
  color: var(--md-default-fg-color--light);
}
/* Difficulty is the one chip that carries a colour - traffic-light order, with
   the tints picked per palette so contrast holds in both light and dark. */
.gallery-badge--beginner     { background: rgba(67,160,71,.16);  color: #2e7d32; }
.gallery-badge--intermediate { background: rgba(251,140,0,.18);  color: #e65100; }
.gallery-badge--advanced     { background: rgba(229,57,53,.16);  color: #c62828; }
[data-md-color-scheme="slate"] .gallery-badge--beginner     { background: rgba(102,187,106,.22); color: #a5d6a7; }
[data-md-color-scheme="slate"] .gallery-badge--intermediate { background: rgba(255,167,38,.22);  color: #ffcc80; }
[data-md-color-scheme="slate"] .gallery-badge--advanced     { background: rgba(239,83,80,.22);   color: #ef9a9a; }
</style>
""".strip()


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        m = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.S)
        if m:
            return text[m.end():]
    return text


def _slugify(title: str) -> str:
    """Approximate Material's toc slugify (lower, drop punctuation, spaces->-)."""
    s = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")


#: Opens a group of sections on the landing page. Goes in the README of the
#: group's *first* section: the title on the marker line, an optional blurb
#: under it. An HTML comment because `mkdocs_gallery.gen_single` strips those
#: before it looks for a section's own title - so the marker cannot be mistaken
#: for one and the left-hand nav keeps naming the section after its `##`.
GROUP_MARKER = re.compile(
    r"<!--\s*gallery-group:[ \t]*(?P<title>.+?)[ \t]*\n(?P<intro>.*?)-->", re.S
)


def _group_key(name: str) -> str:
    """Sections whose folders share a leading number belong together.

    `1a_plotting_general` and `1b_plotting_2d` are one group; `2_morpho` is its
    own. This is the same digit the folders are already sorted by, so a group is
    always contiguous.
    """
    m = re.match(r"(\d+)", name)
    return m.group(1) if m else name


def _pop_group(readme: str):
    """`(group_title, group_intro, readme_without_the_marker)`."""
    m = GROUP_MARKER.search(readme)
    if not m:
        return None, "", readme
    return (
        m.group("title").strip(),
        m.group("intro").strip(),
        (readme[: m.start()] + readme[m.end():]).strip(),
    )


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    return text.replace("`", "").strip()


def _tooltip(intro: str) -> str:
    """Plain, attribute-safe text for the native `title=` tooltip."""
    s = _strip_markdown(re.sub(r"\{\{\s*navis\s*\}\}", "NAVis", intro))
    return html.escape(s, quote=True)


def _desc_html(intro: str, navis_span: str) -> str:
    """Escaped card blurb that keeps the coloured NAVis span."""
    token = "\x00NAVIS\x00"
    s = re.sub(r"\{\{\s*navis\s*\}\}", token, _strip_markdown(intro))
    s = html.escape(s, quote=False)
    return s.replace(token, navis_span)


#: `<!-- difficulty: beginner -->` in a tutorial's docstring. See the module
#: docstring for why it is an HTML comment and where it goes.
DIFFICULTY_MARKER = re.compile(r"<!--\s*difficulty:\s*(?P<level>[\w-]+)\s*-->", re.I)

#: Recognised levels, ascending. Anything else is dropped (with a warning).
DIFFICULTIES = ("beginner", "intermediate", "advanced")


def _difficulty(doc: str, path: Path) -> str | None:
    m = DIFFICULTY_MARKER.search(doc)
    if not m:
        log.warning("%s: no `<!-- difficulty: ... -->` marker in the docstring", path.name)
        return None
    level = m.group("level").lower()
    if level not in DIFFICULTIES:
        log.warning(
            "%s: unknown difficulty %r (expected one of %s)",
            path.name, level, ", ".join(DIFFICULTIES),
        )
        return None
    return level


def _parse_tutorial(path: Path):
    """Return (title, intro, difficulty, tags) from a tutorial's docstring."""
    text = path.read_text(encoding="utf-8")
    # Allow a string prefix (e.g. r""" ... """) before the docstring quotes.
    m = re.match(r'\s*[rRuUbBfF]?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, re.S)
    doc = m.group(1).strip("\n") if m else ""

    difficulty = _difficulty(doc, path)
    # Markers are metadata, not prose - drop them before reading title/intro.
    doc = re.sub(r"<!--.*?-->", "", doc, flags=re.S).strip("\n")
    lines = doc.splitlines()

    title = lines[0].strip() if lines else path.stem
    body = lines[1:]
    if body and re.match(r"^[=\-]+\s*$", body[0]):  # drop the ===/--- underline
        body = body[1:]

    para: list[str] = []
    for line in body:
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    intro = " ".join(para)

    # Optional extension point: `# gallery_tags: beginner, plotting`
    tm = re.search(r"^#\s*gallery_tags:\s*(.+)$", text, re.M)
    tags = [t.strip() for t in tm.group(1).split(",") if t.strip()] if tm else []

    return title, intro, difficulty, tags


def _list_tutorials(folder: Path):
    """Tutorial files directly in `folder`, sorted by name (FileNameSortKey)."""
    return sorted(f for f in folder.glob(TUTORIAL_GLOB))


# --------------------------------------------------------------------------- #
# Card + page assembly
# --------------------------------------------------------------------------- #
def _card(prefix: str, path: Path, gen_dir: Path, navis_span: str) -> str:
    stem = path.stem
    title, intro, difficulty, tags = _parse_tutorial(path)

    thumb = f"{prefix}images/thumb/mkd_glr_{stem}_thumb.png"
    if not (gen_dir / thumb).exists():
        thumb = "../../_static/favicon.png"  # graceful fallback (rarely hit)

    # Difficulty leads, then any `gallery_tags`, then badges derived from
    # structure (remote tutorials pull their data from external services).
    chips = []
    if difficulty:
        chips.append((difficulty, f" gallery-badge--{difficulty}"))
    chips += [(t, "") for t in tags]
    if prefix.startswith("4_remote"):
        chips.append(("remote data", ""))
    badges = ""
    if chips:
        rendered = "".join(
            f'<span class="gallery-badge{cls}">{html.escape(label)}</span>'
            for label, cls in chips
        )
        badges = f'<span class="gallery-badges">{rendered}</span>'

    return (
        f'<a class="gallery-card" href="{prefix}{stem}" title="{_tooltip(intro)}">'
        f'<img class="off-glb" loading="lazy" src="{thumb}" alt="{html.escape(title)}">'
        f'<span class="gallery-card-body">{badges}'
        f'<span class="gallery-card-title">{html.escape(title)}</span>'
        f'<span class="gallery-card-desc">{_desc_html(intro, navis_span)}</span>'
        f"</span></a>"
    )


def build_index(docs_dir: Path, navis_span: str = "NAVis") -> str:
    examples = docs_dir / "examples"
    gen_dir = docs_dir / "generated" / "gallery"

    def expand(md: str) -> str:  # our own tiny {{ navis }} expander
        return re.sub(r"\{\{\s*navis\s*\}\}", navis_span, md)

    # Landing prose + the root "General Tutorials" heading come from the root README.
    root_body = _strip_frontmatter((examples / "README.md").read_text(encoding="utf-8"))
    cut = root_body.find("\n## ")
    if cut == -1:
        landing, root_header = root_body.strip(), "## General Tutorials"
    else:
        landing, root_header = root_body[:cut].strip(), root_body[cut:].strip()

    # mkdocs-gallery is flat: one level of subsections and no deeper - a
    # `GallerySubSection` reports `has_subsections() == False` unconditionally.
    # We own the landing page though, so the grouping happens here. Sections
    # whose folders share a leading number (`1a_...`, `1b_...`) collapse under
    # one heading, taken from the `gallery-group` marker in the first of them.
    #
    # groups: [(title, intro_md, [(section_header_md, files, prefix)])]
    groups = [(None, "", [(root_header, _list_tutorials(examples), "")])]
    for sub in sorted(p for p in examples.iterdir() if p.is_dir() and (p / "README.md").exists()):
        readme = _strip_frontmatter((sub / "README.md").read_text(encoding="utf-8"))
        title, intro, header = _pop_group(readme)
        key = _group_key(sub.name)
        if title is not None or key != _group_key(groups[-1][2][-1][2] or "\x00"):
            groups.append((title, intro, []))
        groups[-1][2].append((header.strip(), _list_tutorials(sub), f"{sub.name}/"))

    # Jump-to-section chips: one per group, so five plotting subsections do not
    # take over the whole bar.
    chips = []
    for title, _, sections in groups:
        for heading in ([f"## {title}"] if title else [s[0] for s in sections]):
            m = re.search(r"^##\s+(.+)$", heading, re.M)
            if m:
                name = m.group(1).strip()
                chips.append(f'<a href="#{_slugify(name)}">{html.escape(name)}</a>')
    jump = '<nav class="gallery-jump">\n' + "\n".join(chips) + "\n</nav>"

    parts = [STYLE, expand(landing), jump]
    for title, intro, sections in groups:
        if title:
            parts.append(f"## {title}")
            if intro:
                parts.append(expand(intro))
        for header, files, prefix in sections:
            # inside a group the sections' own `##` headings drop a level
            parts.append(expand(re.sub(r"^##(?= )", "###", header, flags=re.M) if title else header))
            cards = "\n".join(_card(prefix, f, gen_dir, navis_span) for f in files)
            parts.append(f'<div class="gallery-grid">\n{cards}\n</div>')

    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# mkdocs hook entry point
# --------------------------------------------------------------------------- #
def on_page_markdown(markdown: str, *, page, config, files):
    if page.file.src_uri != GALLERY_INDEX:
        return markdown
    try:
        navis_span = (config.get("extra") or {}).get("navis", "NAVis")
        return build_index(Path(config["docs_dir"]), navis_span)
    except Exception:  # never let the prototype break the whole build
        log.exception("custom gallery index hook failed - falling back to stock index")
        return markdown
