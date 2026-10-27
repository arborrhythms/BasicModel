"""Every relative Markdown link under ``doc/``, ``README.md`` and ``todo.md`` resolves.

Documentation is part of each milestone (mathematical-thinking plan,
Phase 0): a spec that links a plan, a plan that links a test, or a README
row that links a document must point at a file that exists. External links
(``http``/``https``/``mailto``) and pure in-page anchors are not checked.
"""
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _markdown_files():
    doc = _ROOT / "doc"
    # Receipts retain extracted executable snapshots as historical evidence.
    # Their copied root README refers to a documentation tree deliberately
    # absent from the source archive. Audit receipt prose, not those copies.
    files = sorted(path for path in doc.rglob("*.md")
                   if not any((parent / "bin/Models.py").is_file()
                              and (parent / "data/model.xml").is_file()
                              for parent in path.parents if doc in parent.parents))
    for name in ("README.md", "todo.md"):
        path = _ROOT / name
        if path.exists():
            files.append(path)
    return files


def _relative_targets(text):
    for match in _LINK.finditer(text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        # ``<...>`` autolink-style targets and templated placeholders are
        # not file references.
        if target.startswith("<") or "{" in target:
            continue
        target = target.split("#", 1)[0]
        # ``path:line`` clickable code references.
        yield unquote(re.sub(r":\d+$", "", target))


@pytest.mark.parametrize("path", _markdown_files(),
                         ids=lambda p: str(p.relative_to(_ROOT)))
def test_relative_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    missing = []
    # These are byte-preserved snapshots of documents, whose links retain
    # the original document's home. Audit every target without rewriting
    # historical evidence or excluding the archived documents.
    origins = {
        **{f"doc/benchmarks/2026-10-25-item3-0-retire-mode/repair-04/development/upstream/{name}.md": "doc/FutureWork.md"
           for name in ("base", "local", "merged", "remote")},
        **{f"doc/benchmarks/2026-10-25-item3-0-retire-mode/repair-04/publication/FutureWork.{name}.md": "doc/FutureWork.md"
           for name in ("combined", "conflict")},
        **{f"doc/benchmarks/2026-10-25-item3-0-stm-chunks/{name}": "doc/plans/2026-10-25-item3-0-step3-stm-chunks.md"
           for name in ("repair-01/spec-as-received.md", "step3-spec-as-received.md")},
    }
    original = _ROOT / origins.get(str(path.relative_to(_ROOT)), str(path.relative_to(_ROOT)))
    for target in _relative_targets(text):
        if not target:
            continue
        resolved = (original.parent / target).resolve()
        if not resolved.exists():
            missing.append(target)
    assert not missing, f"{path.relative_to(_ROOT)}: unresolved {missing}"
