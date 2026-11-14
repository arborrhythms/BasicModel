"""Relative links in non-ignored repository Markdown resolve from their document.

Check ``doc/**/*.md``, ``README.md`` and ``todo.md``, including uncommitted
documents, except paths reported by git check-ignore. Local ignored receipts
retain their historical bytes. External links (``http``/``https``/``mailto``)
and pure in-page anchors are not checked.
"""
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _markdown_files():
    files = sorted(path for path in (_ROOT / "doc").rglob("*.md") if path.is_file())
    for name in ("README.md", "todo.md"):
        path = _ROOT / name
        if path.exists():
            files.append(path)
    ignored = subprocess.run(["git", "-C", str(_ROOT), "check-ignore", "--stdin", "-z"],
        input='\0'.join(str(path.resolve()) for path in files)+'\0',
        capture_output=True, text=True)
    if ignored.returncode not in (0, 1):
        raise RuntimeError('Cannot determine ignored document paths: '+ignored.stderr)
    excluded = set(ignored.stdout.rstrip('\0').split('\0'))
    return [path for path in files if str(path.resolve()) not in excluded]


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
