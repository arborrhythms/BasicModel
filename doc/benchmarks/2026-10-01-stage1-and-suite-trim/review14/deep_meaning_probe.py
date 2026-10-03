"""LM_5M's second host traversal: a deep selected tree must not overflow Python."""
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'bin'))
import torch
from Language import LanguageSpace

def check(words=1200):
    rows = [(0, -1, 0)]
    for word in range(1, words):
        rows.extend(((0, -1, word), (1, 0, -1)))
    entry = SimpleNamespace(actions=torch.tensor(rows), leaves=torch.ones(words, 2),
                            concept_ids=torch.arange(1, words+1))
    owner = SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='sum')],
                            _compose_unary_rules=[])
    registry = SimpleNamespace(form=lambda *a, **k: None)
    # A sum-only selected program makes no grammatical claim. Recovery is
    # bounded, but projecting its structure must accept the configured length.
    assert LanguageSpace.program_meaning(owner, entry, registry) is None

if __name__ == '__main__':
    if '--before' in sys.argv:
        import ast, zipfile, Language
        with zipfile.ZipFile(Path(__file__).with_name('before.zip')) as archive:
            tree = ast.parse(archive.read('bin/Language.py'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LanguageSpace')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'program_meaning')
        namespace = dict(vars(Language))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<saved Language.py>', 'exec'), namespace)
        LanguageSpace.program_meaning = namespace['program_meaning']
    check()
    print('PASS: 1,200-word selected tree, no fabricated meaning')
