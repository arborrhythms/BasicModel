"""Supplement the unchanged bounded runner's source manifest with fixture inputs."""
import hashlib
from pathlib import Path


def supporting_inputs(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / 'test/fixtures').rglob('*'))
        if path.is_file() and path.suffix not in ('.py', '.pyc')
        and '__pycache__' not in path.parts
    }
