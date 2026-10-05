"""Replay the four saved §22 roots; no training or initialization is repeated."""
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch
import pytest


@pytest.mark.parametrize('compiled', [False, True])
def test_saved_roots_follow_operand_unaries(monkeypatch, compiled):
    import Models
    from Language import LanguageSpace, ConjunctionLayer, NotLayer
    folder = Path(__file__).resolve().parents[1] / 'doc/benchmarks/2026-10-03-item6-9-free-readback/xor-ownership'
    saved = torch.load(folder/'geometry-start-stage-1.pt', weights_only=True)
    codes = {int(row): code for row, code in zip(saved['rows'], saved['codes'])}
    pairs = [(0, 1), (0, 2), (4, 1), (4, 2)]
    basis = torch.stack([torch.stack([codes[a], codes[b]]) for a,b in pairs])
    roots = torch.tensor(json.loads((folder/'geometry-start.json').read_text())['roots']['values'])
    conjunction, negation = ConjunctionLayer(), NotLayer()
    language = SimpleNamespace()
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'reverse_unary_step'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    language.local_op_from_rule_ids = LanguageSpace.local_op_from_rule_ids
    # The saved roots encode two independent negation choices. Check the
    # exact inverse law from the same saved candidate bank; the decoder no
    # longer reads a compose journal to discover these choices.
    left_neg=torch.tensor([False,False,True,True])[:,None]
    root_neg=torch.tensor([False,True,False,True])[:,None]
    def reconstruct(root,candidates):
        parent=torch.where(root_neg,-root,root)
        left=torch.where(left_neg,-candidates[:,0],candidates[:,0])
        bank=torch.stack((left,candidates[:,1]),1)
        a,b,missing=language.reverse_binary_step(parent,torch.zeros(4,dtype=torch.long),
            torch.ones(4,dtype=torch.bool),ops=[conjunction],basis=bank,
            basis_valid=torch.ones(4,2,dtype=torch.bool),free=True,return_status=True)
        recovered=torch.stack((torch.where(left_neg,-a,a),b),1)
        return recovered,None,None,missing,None
    inverse = torch.compile(reconstruct, backend='inductor', fullgraph=True) if compiled else reconstruct
    recovered, _, _, unavailable, _ = inverse(roots, basis)
    assert not unavailable.any()
    torch.testing.assert_close(recovered, basis)
    left = torch.where(torch.tensor([False,False,True,True])[:,None], -recovered[:,0], recovered[:,0])
    composed = conjunction.compose(left, recovered[:,1])
    composed = torch.where(torch.tensor([False,True,False,True])[:,None], -composed, composed)
    torch.testing.assert_close(composed, roots)
