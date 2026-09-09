"""Every relative Markdown link under ``doc/`` and in ``README.md`` resolves.

Documentation is part of each milestone (mathematical-thinking plan,
Phase 0): a spec that links a plan, a plan that links a test, or a README
row that links a document must point at a file that exists. External links
(``http``/``https``/``mailto``) and pure in-page anchors are not checked.
"""
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _markdown_files():
    files = sorted((_ROOT / "doc").rglob("*.md"))
    readme = _ROOT / "README.md"
    if readme.exists():
        files.append(readme)
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
        yield re.sub(r":\d+$", "", target)


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
