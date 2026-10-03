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
    binary = SimpleNamespace(ops=[conjunction])
    unary = SimpleNamespace(unary_ops=[negation])
    language._tree_layer = lambda arity: binary if arity == 2 else unary
    B, W, D, cap = 4, 2, 10, 3
    steps = W*3+(W+1)*2*cap
    rule_ids = torch.zeros(B, steps, dtype=torch.long)
    arities = torch.zeros_like(rule_ids)
    arities[:, 3] = 2
    arities[2:, 0] = 1
    arities[[1,3], 4] = 1
    rule_ids[arities == 1] = 1
    trace = SimpleNamespace(choices=lambda: (rule_ids, arities, arities > 0),
        _choice_positions=torch.zeros_like(rule_ids),
        rule_map=lambda arity: torch.tensor([0 if arity == 2 else 1]))
    owner = SimpleNamespace(inputSpace=SimpleNamespace(_word_active_mask=torch.ones(B,W,dtype=torch.bool)),
        languageSpace=language, conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=cap)),
        _reconstruction_stack=lambda: trace, reconstruction_basis_limit=2)
    owner._byte_word_cost = MethodType(Models.BasicModel._byte_word_cost, owner)
    owner._tensor_write_word_column = Models.BasicModel._tensor_write_word_column
    def eager_loop(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    if not compiled:
        monkeypatch.setattr(Models, '_reconstruction_while_loop', eager_loop)
    def reconstruct(root, candidates):
        return Models.BasicModel._reconstruct_sentences(
            owner, root, candidates, root[:, None],
            end_slots=torch.cat((root[:,None],root.new_zeros(B,2,D)),1),
            end_depth=torch.ones(B,dtype=torch.long),
            candidate_basis=(candidates,torch.ones(B,2,dtype=torch.bool)), keep_ideas=True)
    inverse = torch.compile(reconstruct, backend='inductor', fullgraph=True) if compiled else reconstruct
    recovered, _, _, unavailable, _ = inverse(roots, basis)
    assert not unavailable.any()
    torch.testing.assert_close(recovered, basis)
    left = torch.where(torch.tensor([False,False,True,True])[:,None], -recovered[:,0], recovered[:,0])
    composed = conjunction.compose(left, recovered[:,1])
    composed = torch.where(torch.tensor([False,True,False,True])[:,None], -composed, composed)
    torch.testing.assert_close(composed, roots)
