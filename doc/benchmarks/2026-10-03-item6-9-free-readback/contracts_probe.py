"""Pre-repair behavioral probes for the decisions in plan §§21–22."""
from dataclasses import fields
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]


def test_conjunction_carries_product_magnitude_and_identity():
    from Language import ConjunctionLayer
    left = torch.tensor([[.2, -.7, .4]])
    right = torch.tensor([[.8, .3, -.5]])
    expected = (left.norm(dim=-1, keepdim=True) * right.norm(dim=-1, keepdim=True)
                * torch.nn.functional.normalize(left * right, dim=-1))
    torch.testing.assert_close(ConjunctionLayer().compose(left, right), expected)


def test_disjunction_is_the_mean():
    from Language import DisjunctionLayer
    left = torch.tensor([[.2, -.7, .4]])
    right = torch.tensor([[.8, .3, -.5]])
    torch.testing.assert_close(DisjunctionLayer().compose(left, right), (left+right)/2)


def test_min_and_max_remain_named_catalogue_operators():
    import Language
    assert hasattr(Language, 'MinLayer') and hasattr(Language, 'MaxLayer')


def test_understanding_does_not_carry_witness_offsets():
    from SentenceUnderstanding import SentenceUnderstanding
    assert 'witness_offsets' not in {f.name for f in fields(SentenceUnderstanding)}


def test_reconstruction_optimizer_is_gradient_descent_with_momentum(monkeypatch):
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, _ = _fresh_model(str(ROOT/'data/XOR_grammar.xml'))
    try:
        optimizer = model.getOptimizer(lr=.01)
        writers = model.objective_parameter_groups(optimizer)
        children = getattr(optimizer, 'optimizers', [optimizer])
        for owner in ('reconstruction', 'output'):
            for parameter in writers[owner]:
                child = next(o for o in children if any(parameter is p
                             for g in o.param_groups for p in g['params']))
                inner = getattr(child, 'inner', child)
                if owner == 'reconstruction':
                    assert isinstance(inner, torch.optim.SGD) or type(inner).__name__ == '_RowLocalSGD'
                    assert all(g['momentum'] > 0 for g in inner.param_groups)
                else:
                    assert isinstance(inner, (torch.optim.Adam, torch.optim.SparseAdam)) or type(inner).__name__ == '_RowLocalAdam'
    finally:
        model.End()
