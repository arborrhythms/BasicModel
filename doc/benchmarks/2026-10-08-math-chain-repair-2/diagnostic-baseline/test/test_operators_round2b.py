"""Reconstruction keep, unbiased reader exposure, and inherited form order."""
from types import SimpleNamespace
import torch


def test_reader_mean_for_compose_kept_for_narrowing_and_one_owned_step():
    from Layers import Error
    from SentenceCredit import reader_weights, reader_costs
    from ObjectiveOwnership import backward_owned
    active = torch.tensor([True, True, True, True, False])
    parts = torch.zeros(5, 2, 3)
    parts[:, 0, 0] = torch.tensor([0., 1., 0., 1., 1.])
    parts[:, 1, 0] = torch.tensor([1., 0., 0., 0., 0.])
    draw = dict(compose_round=torch.tensor([0, 0, -1, -1, -1]))
    weights = reader_weights(parts, active, draw)
    torch.testing.assert_close(weights, torch.tensor([[.5,.5],[.5,.5],[1.,0.],[0.,1.],[0.,0.]]))
    parameter = torch.nn.Parameter(torch.tensor(2.))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    roots = [torch.arange(1., 6., requires_grad=True), torch.arange(6., 11., requires_grad=True)]
    registries = []
    for root in roots:
        registry = Error(row_mask=active)
        registry.error('answer', (parameter * root.detach()).square(), 1., category='output')
        registries.append(registry)
    costs = reader_costs(registries, weights)
    expected = sum((parameter * root.detach()).square() * weights[:, i] for i, root in enumerate(roots)).sum() / active.sum()
    torch.testing.assert_close(costs['output'], expected)
    backward_owned({}, {'output': (parameter,)})
    optimizer.step()
    assert parameter not in optimizer.state
    backward_owned(costs, {'output': (parameter,)})
    optimizer.step()
    assert optimizer.state[parameter]['step'] == 1
    assert all(root.grad is None for root in roots)


def test_pole_handoff_preserves_both_magnitudes_independently_of_presence():
    from ModelAttention import reference_evidence
    presence = torch.tensor([[[.3], [.6], [.8], [.4]]], requires_grad=True)
    pair = torch.tensor([[[0., 1.], [1., 0.], [.5, .5], [1., 0.]]])
    changed = torch.tensor([[True, True, True, False]])
    model = SimpleNamespace(_attention_poles=pair,
                            _attention_words=SimpleNamespace(pole_changes=changed))
    evidence = reference_evidence(model, presence, 'commit_word_reference_slab:whole_slab')
    torch.testing.assert_close(evidence, pair, rtol=0, atol=0)
    assert not evidence.requires_grad
    model._attention_poles = None
    native = torch.stack((presence.squeeze(-1).clamp(0, 1),
                         torch.zeros_like(presence.squeeze(-1))), -1)
    torch.testing.assert_close(reference_evidence(model, presence,
        'commit_word_reference_slab:per_word'), native, rtol=0, atol=0)


def test_inherited_parts_form_partial_order_and_before_after_audit():
    from test_review13_subspaces import fixture, word
    cs, cb, ps, _ = fixture()
    with torch.no_grad():
        ps.W[4] = torch.tensor([.2, .2, .2])
        ps.W[7] = torch.tensor([.8, .3, .2])
        ps.W[8] = torch.tensor([.3, .8, .2])
    a = word(cs, [(4, 1.)])
    ab = word(cs, [(4, 1.), (7, 1.)])
    ac = word(cs, [(4, 1.), (8, 1.)])
    codes = dict(zip((a, ab, ac), cb.lookup_rows(torch.tensor([a, ab, ac]))))
    assert (codes[a][:3] <= codes[ab][:3]).all()
    assert (codes[a][:3] <= codes[ac][:3]).all()
    assert not (codes[ab][:3] <= codes[ac][:3]).all()
    assert not (codes[ac][:3] <= codes[ab][:3]).all()
    audit = cb.mereology.containment_audit()
    assert audit['before'] == audit['after'] == dict(pairs=2, violating_pairs=0, coordinates=0, largest=0.)
    displaced = {row: code.clone() for row, code in codes.items()}
    displaced[a][0] = 1.
    audit = cb.mereology.containment_audit(before=displaced, after=codes)
    assert audit['before']['violating_pairs'] == 2
    assert audit['before']['largest'] > .6 and audit['after']['largest'] == 0
