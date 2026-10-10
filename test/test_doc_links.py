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
    for target in _relative_targets(text):
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            missing.append(target)
    assert not missing, f"{path.relative_to(_ROOT)}: unresolved {missing}"
