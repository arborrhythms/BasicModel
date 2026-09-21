"""List explicit test RNG seeds for review, without claiming seed independence."""
import ast
import json
from pathlib import Path


def inventory():
    entries = []
    for path in sorted(Path('test').rglob('test_*.py')):
        tree = ast.parse(path.read_text())
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if not (name.endswith('.manual_seed') or name in
                    ('random.seed', 'np.random.seed', 'numpy.random.seed', 'random.Random', 'np.random.default_rng')):
                continue
            owner = node
            while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            entries.append({'file': str(path), 'line': node.lineno,
                            'function': getattr(owner, 'name', '<module>'),
                            'call': ast.unparse(node)})
    return entries


if __name__ == '__main__':
    print(json.dumps(inventory(), indent=2))
