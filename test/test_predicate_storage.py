"""Compact host predicates keep their original addresses and exact boundaries."""
import torch
import pytest
from Spaces import _predicate_unit_spans
from test_wholespace_property_migration import _small_property_model


def fixture(tmp_path, mode):
    owner = _small_property_model(tmp_path).wholeSpace
    primitives = owner.subspace.what.primitive_properties
    with torch.no_grad():
        primitives.members.zero_()
        primitives.members[1, [49, 50]] = 1.
        primitives.members[5, [50, 51]] = 1.
        owner.begins_weight.fill_(-8.)
        owner.ends_weight.fill_(-8.)
        owner.atom_level.fill_(-8.)
        if mode == 'normal':
            owner.begins_weight[257] = 8.
            owner.ends_weight[261] = 8.
        elif mode == 'absent':
            owner.begins_weight[260] = 8.
    idx = torch.tensor([[49, 49, 50, 51, 0], [50, 32, 49, 51, 0]])
    dense = torch.zeros(*idx.shape, owner._predicate_column_count(), dtype=torch.bool)
    dense.scatter_(2, idx[..., None], True)
    lookup = primitives.coefficients().detach().t() > .5
    lookup[0] = False
    dense[..., 256:] = lookup[idx]
    masks = owner._predicate_masks()
    effective = list(masks)
    if not bool(effective[0].any() or effective[1].any() or effective[2].any()):
        effective[2] = torch.ones_like(effective[2])
    expected = _predicate_unit_spans(dense, *effective)
    return owner, idx, dense, masks, expected


@pytest.mark.parametrize('mode', ['normal', 'cold', 'absent'])
def test_compact_predicates_preserve_spans_and_global_columns(tmp_path, mode):
    owner, idx, dense, masks, expected = fixture(tmp_path, mode)
    actual, packet = owner._unit_tiling_from_predicates(idx)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for row in range(idx.shape[0]):
        wanted = [(int(a), int(z)) for a, z in expected[row].tolist() if z > a]
        assert owner.unit_spans_of_bytes(bytes(idx[row].tolist())) == wanted
    slab, columns = packet if isinstance(packet, tuple) else (packet, torch.arange(dense.shape[-1]))
    present = dense.any((0, 1)).nonzero().flatten()
    assert torch.equal(columns, present), (columns, present)
    torch.testing.assert_close(slab, dense[..., present], atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['normal', 'cold', 'absent'])
def test_compact_predicates_preserve_boundary_learning_evidence(tmp_path, mode):
    owner, idx, dense, masks, current = fixture(tmp_path, mode)
    candidates = [(('current',), current)]
    for c in dense.any((0, 1)).nonzero().flatten().tolist():
        if bool(masks[3][c]):
            continue
        for i, move in enumerate(('begins', 'ends', 'atom')):
            changed = [v.clone() for v in masks]
            changed[i][c] = True
            candidates.append(((move, c), _predicate_unit_spans(dense, *changed)))
    expected = {}
    vals = idx.tolist()
    for key, spans in candidates:
        rec = dict(surfaces={}, units=0, presentations=0)
        for b, row in enumerate(spans.tolist()):
            rec['presentations'] += 1
            for a, z in row:
                if z > a:
                    surface = bytes(vals[b][a:z])
                    rec['surfaces'][surface] = rec['surfaces'].get(surface, 0) + 1
                    rec['units'] += 1
        expected[key] = rec
    actual, packet = owner._unit_tiling_from_predicates(idx)
    owner._observe_candidate_tilings(idx, packet, actual)
    assert owner._boundary_evidence == expected
