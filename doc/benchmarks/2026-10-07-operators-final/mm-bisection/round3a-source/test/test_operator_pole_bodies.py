"""Execute the preserved landing/candidate pole bodies on the same values."""
import ast
import json
from pathlib import Path
import torch


def test_saved_old_new_pole_bodies_and_current_source_agree():
    import Language
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root/'doc/benchmarks/2026-10-05-operators-update/pole-bodies.json').read_text())
    source = (root/'bin/Language.py').read_text()
    current = {node.name: ast.get_source_segment(source, node)
               for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    poles = torch.tensor([[.8, .1], [.1, .8], [.5, .5], [0., 0.]])
    other = torch.tensor([[.3, .6], [.3, .6], [.3, .6], [.3, .6]])
    implementations = {}
    for phase, bodies in saved.items():
        namespace = dict(Language.__dict__)
        for name, body in bodies.items():
            if phase == 'new':
                assert body == current[name], 'refresh the saved body when production changes'
            exec(compile(body, f'saved-{phase}-{name}', 'exec'), namespace)
        implementations[phase] = namespace
    old, new = implementations['old'], implementations['new']
    # Withdrawal was already repaired at the landing; both saved bodies
    # demonstrate zero expressed evidence, with the opposite pole preserved.
    for implementation in (old, new):
        result = implementation['NonLayer'](representation='poles')(poles)
        torch.testing.assert_close(result[:, 0], torch.zeros(4))
        torch.testing.assert_close(result[:, 1], poles[:, 1])
        torch.testing.assert_close(implementation['NotLayer'](representation='poles')(poles), poles.flip(-1))
    assert torch.equal(old['NotLayer']()(poles), -poles)
    assert torch.equal(new['NotLayer']()(poles), poles)
    expected = torch.stack((torch.minimum(poles[:, 0], other[:, 0]),
                            torch.maximum(poles[:, 1], other[:, 1])), -1)
    assert not torch.allclose(old['ConjunctionLayer']()(poles, other), expected)
    torch.testing.assert_close(new['ConjunctionLayer'](representation='poles')(poles, other), expected)
