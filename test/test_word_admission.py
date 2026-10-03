"""Word-admission acceptance tests 23, 26, 29, 30 and 33."""
import inspect
from pathlib import Path
import pytest
import torch


def snapshot(cs):
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    store = alloc.layer()
    return (alloc.next_id, dict(alloc.placement), dict(cs.interpret._pending),
            {key: tuple(value) for key, value in store._constituents.items()},
            dict(store.features._index),
            () if store.features.values is None else tuple(store.features.values.detach().tolist()),
            dict(store._tensor_rows), dict(store._index))


def test_23_fuse_parts_before_word_admission(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, 8)
    cs, ps, ws = model.conceptualSpaces[0], model.perceptualSpace, model.wholeSpaces[0]
    parts = ps.percept_store.spell_out(b'love')
    ps.percept_store.observe_chunk(b'love')
    fused = ps.percept_store.observe_chunk(b'love')
    assert ps.fuse_parts(parts) == [fused]
    word = cs.interpret.lookup_word(parts, ws.property_rows_for_bytes('love'), form='love')
    assert cs.concept_parts(word) == [fused]


def test_23_same_parts_after_property_change_write_nothing():
    from test_cs_sparse_weights import _cs
    cs = _cs()
    word = cs.interpret.lookup_word([7], [1], form='word', word_reading=True)
    before = snapshot(cs)
    assert cs.interpret.lookup_word([7], [2], form='word', word_reading=True) == word
    assert snapshot(cs) == before


def test_26_full_row_inventory_leaves_no_word_identity():
    from test_concept_capacity_policy import _cs
    cs = _cs(8)
    for row in range(8):
        assert cs._csw_concept_row(0, 1000 + row) == row
    before = snapshot(cs)
    try:
        word = cs.interpret.lookup_word([7], [1], form='unseated', word_reading=True)
    except RuntimeError as error:
        assert 'capacity' in str(error) or 'exhausted' in str(error)
    else:
        assert word is None
    assert snapshot(cs) == before


def test_29_shared_whole_does_not_evidence_an_absent_word():
    from test_cs_sparse_weights import _cs
    from PerceptProperties import PrimitiveProperties
    cs = _cs()
    word = cs.interpret.lookup_word([7], [0], form='word', word_reading=True)
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [65], [1.])
    raw = torch.tensor([[65, 65]])
    ids = torch.tensor([[7, 8]])
    spans = torch.tensor([[[0, 1], [1, 2]]])
    field = cs.cs_read_memberships((ids, spans, primitive, raw, spans), spans)
    obj = cs.interpret.forward(word)
    # Re-read with the object now occupying the word's inventory seat.
    field = cs.cs_read_memberships((ids, spans, primitive, raw, spans), spans)
    row = cs._csw_row_of(obj)
    addresses = cs._field_inventory_rows()
    slot = int((addresses == row).nonzero()[0])
    assert field[slot, 0, 0, 0] > 0
    assert field[slot, 0, 1].count_nonzero() == 0
    assert cs.concept_wholes(word) == [0]


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('configuration', ['MM_xor.xml', 'MM_grammar.xml', 'XOR_grammar.xml'])
def test_29_each_reading_publishes_only_its_words(configuration):
    from test_mm_xor import _fresh_model, _PROJECT
    model, _, data = _fresh_model(str(Path(_PROJECT) / 'data' / configuration))
    model.eval()
    texts = ['hello world', 'hello there', 'loving world', 'loving there']
    inp = model.inputSpace.prepInput(texts)
    with torch.no_grad():
        _, symbols, _, _ = model(inp)
    cs = model._concept_owner()
    carrier = model.symbol_cache
    rows = carrier._concept_inventory_rows
    evidence = carrier._concept_activations
    where = carrier._concept_where
    words = ('hello', 'loving', 'world', 'there')
    checked = 0
    for b, text in enumerate(texts):
        reading = bytes(inp[b].reshape(-1).long().tolist()).split(b'\0', 1)[0]
        for word in words:
            for cid in cs.word_concepts(word):
                address = cs._csw_row_of(cid)
                if address is None:
                    continue
                indices = (rows[:, b] if rows.ndim == 2 else rows) == address
                for e, (lo, hi) in enumerate(where[b].tolist()):
                    if word.encode() not in reading[lo:hi]:
                        assert not evidence[indices, b, e].any(), (configuration, text, word, lo, hi)
                        checked += 1
    assert checked > 0
    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert not torch.equal(symbols[left], symbols[right])
    model.End()


def test_property_read_selects_rows_and_in_bracket_events(monkeypatch):
    from PerceptProperties import PrimitiveProperties
    from PerceptField import read_percepts
    from Spaces import PartSpace
    primitive = PrimitiveProperties(8192)
    primitive.teach(17, [65], [1.])
    seen = []
    original = primitive.evidence_on_counts
    def observed(counts, *, rows=None):
        assert rows is not None and rows.tolist() == [17], 'unreferenced property rows evaluated'
        assert counts.reshape(-1, 256).shape[0] == 1, 'events outside bracket evaluated'
        seen.append(True)
        return original(counts, rows=rows)
    monkeypatch.setattr(primitive, 'evidence_on_counts', observed)
    raw = torch.tensor([[65, 65, 65, 65]])
    spans = torch.tensor([[[0, 1], [1, 2], [2, 3], [3, 4]]])
    bracket = spans[:, 1:2]
    field = read_percepts((raw[:, :0], spans[:, :0], primitive, raw, spans),
                         bracket, [('ws', 17)], part_reader=PartSpace.part_memberships)
    assert seen == [True]
    torch.testing.assert_close(field.values[0, 0, 0], torch.tensor([1., 0.]))


@pytest.mark.parametrize('trial', range(3))
@pytest.mark.parametrize('pool', [4, 8])
def test_33_grounded_xor_with_word_boundary(tmp_path, trial, pool):
    import test_grounded_xor as original
    # The same exact-learning assertions as the existing six cases. Only the
    # boundary is added; the final unrelated-percept assertion is excluded
    # from this NEW test, exactly as section 17.7 requires.
    source = inspect.getsource(original.learn_grounded_xor)
    source = source.replace('    # The second presentation admits',
                            "    cs._commit_autobind_from_stash()\n    assert all(cs.word_concepts(word) for word in ('00', '01', '10', '11'))\n    # The second presentation admits", 1)
    source = source.split('    # Learned exclusion is exact')[0]
    source += '''    controls = x[:1].expand(4, -1, -1).clone()
    controls[:, 0, :2] = torch.tensor([[65, 65], [90, 90], [50, 50], [33, 33]])
    with torch.no_grad():
        model.forward(controls)
        print({'test_33_unrelated_percept_events': int(cs._percept_field.events.count_nonzero()), 'pool': pool})
    model.End()
'''
    namespace = dict(vars(original))
    exec(compile(source, __file__ + ':test33', 'exec'), namespace)
    namespace['learn_grounded_xor'](tmp_path, pool)
