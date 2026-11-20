"""Word-admission acceptance tests 23, 26, 29, 30 and 33."""
import inspect
from pathlib import Path
import pytest
import torch
from nomination_fixtures import recurrent_word, nominated_forms, recurrent_interpretation, readings_before_nomination


def snapshot(cs):
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    store = alloc.layer()
    return (alloc.next_id, dict(alloc.symbols), dict(cs.interpret._pending),
            {key: tuple(value) for key, value in store._constituents.items()},
            dict(alloc.order_zero_members),
            dict(store._tensor_rows), dict(store._index))


def test_23_fuse_parts_before_word_admission(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path, 8)
    cs, ps, ws = model.conceptualSpaces[0], model.perceptualSpace, model.wholeSpace
    parts = ps.percept_store.spell_out(b'love')
    ps.percept_store.observe_chunk(b'love')
    fused = ps.percept_store.observe_chunk(b'love')
    assert ps.fuse_parts(parts) == [fused]
    word = recurrent_word(cs.interpret, parts, ws.property_rows_for_bytes('love'), form='love')
    assert set(cs.concept_parts(word)) == set(ps.percept_store.identity.admit(b'love'))
    cs.interpret.forward(word)
    assert cs.interpret.word_for_form('love') == word
    assert cs.interpret._pending[word]['identity_key'] == ps.percept_store.identity.key(
        ps.percept_store.identity.form(b'love')).hex()
    assert not cs.definitions.objects(word)


def test_23_same_parts_after_property_change_write_nothing():
    from test_cs_sparse_weights import _cs
    cs = _cs()
    word = recurrent_word(cs.interpret, [7], [1], form='word', word_reading=True)
    before = snapshot(cs)
    assert recurrent_word(cs.interpret, [7], [2], form='word', word_reading=True) == word
    assert snapshot(cs) == before


def test_26_full_row_inventory_leaves_no_word_identity():
    from test_concept_capacity_policy import _cs
    cs = _cs(8)
    from order_fixtures import force_order
    # Word admission needs a seat in order zero; higher blocks cannot spill
    # into it. Fill its constructed capacity, keeping the total eight rows.
    for row in range(cs.order_slice(0)[1]):
        force_order(cs, 1000 + row, 0)
        assert cs._csw_concept_row(0, 1000 + row) == row
    before = snapshot(cs)
    try:
        word = recurrent_word(cs.interpret, [7], [1], form='unseated', word_reading=True)
    except RuntimeError as error:
        assert 'capacity' in str(error) or 'exhausted' in str(error)
    else:
        assert word is None
    assert snapshot(cs) == before


def test_29_shared_whole_does_not_evidence_an_absent_word():
    from test_cs_sparse_weights import _cs
    from PerceptProperties import PrimitiveProperties
    cs = _cs()
    word = recurrent_word(cs.interpret, [7], [0], form='word', word_reading=True)
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [65], [1.])
    raw = torch.tensor([[65, 65]])
    ids = torch.tensor([[7, 8]])
    spans = torch.tensor([[[0, 1], [1, 2]]])
    field = cs.cs_read_memberships((ids, spans, primitive, raw, spans), spans)
    obj = cs.interpret.forward(word)
    # Re-read with the object now occupying the word's inventory seat.
    field = cs.cs_read_memberships((ids, spans, primitive, raw, spans), spans)
    row = cs._concept_row_of(obj)
    addresses = cs._field_inventory_rows()
    slot = int((addresses == row).nonzero()[0])
    assert field[slot, 0, 0, 0] > 0
    assert field[slot, 0, 1].count_nonzero() == 0
    assert cs.concept_wholes(word) == [0]


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('configuration', ['MM_text.xml', 'MM_grammar.xml', 'XOR_grammar.xml'])
def test_29_each_reading_publishes_only_its_words(configuration, monkeypatch):
    from reading_fixtures import keep_existing_references
    keep_existing_references(monkeypatch)
    from test_mm_xor import _fresh_model, _PROJECT
    from contextlib import nullcontext
    from reading_fixtures import field_text_config
    # This parameter tests the former field-only word read at the same
    # dimensions; §3.1 now requires its operations to declare that path.
    path = str(Path(_PROJECT) / 'data' / configuration)
    context = field_text_config(examine=False) if configuration == 'MM_text.xml' else nullcontext(path)
    with context as path:
        model, _, data = _fresh_model(path)
    model.eval()
    texts = ['hello world', 'hello there', 'loving world', 'loving there']
    inp = model.inputSpace.prepInput(texts)
    # A form is a word from the reading in which MTM nominates it.
    readings_before_nomination(model, lambda: model(inp))
    with torch.no_grad():
        _, symbols, _, _ = model(inp)
    # §68.73: these are the starting source's staged brackets, including
    # the word-major separator slot and the whole-slab word ordinals.
    expected = torch.zeros_like(model._attention_spans)
    second = 1 if configuration == 'MM_text.xml' else 2
    for b in range(4):
        length = 5 if b < 2 else 6
        expected[b, 0] = expected.new_tensor((0, length))
        expected[b, second] = expected.new_tensor((length + 1, length + 6))
    torch.testing.assert_close(model._attention_spans, expected)
    cs = model._concept_owner()
    carrier = model.symbol_cache
    rows = carrier._concept_inventory_rows
    evidence = carrier._concept_activations
    where = carrier._concept_where
    words = ('hello', 'loving', 'world', 'there')
    # A concept of order zero reads every reading by its entries in the
    # order-zero Sigma, on the rows' signed values (rule 62). Where its word
    # is absent, what is published for it is what its entries read of the
    # rows present there, and no more; untaught, it has one entry, on its
    # own row, so it reads zero. A concept above order zero publishes
    # nothing there.
    perception, sigma = cs._percept_field, cs.order_zero
    n0 = sigma.nOutput
    inputs = [key[1] for key in perception.keys]
    checked = 0
    for b, text in enumerate(texts):
        reading = bytes(inp[b].reshape(-1).long().tolist()).split(b'\0', 1)[0]
        for word in words:
            for cid in cs.word_concepts(word):
                address = cs._concept_row_of(cid)
                if address is None:
                    continue
                indices = (rows[:, b] if rows.ndim == 2 else rows) == address
                if address >= n0:
                    for e, (lo, hi) in enumerate(where[b].tolist()):
                        if word.encode() not in reading[lo:hi]:
                            assert not evidence[indices, b, e].any(), (configuration, text, word, lo, hi)
                            checked += 1
                    continue
                assert [(w, n) for _, w, n in sigma.entries(address)] == [(1., False)], 'an untaught concept reads its own row'
                for e, (lo, hi) in enumerate(where[b].tolist()):
                    if word.encode() not in reading[lo:hi]:
                        presence = perception.values[:, b:b + 1, e:e + 1].to(evidence)
                        expected = cs.order_zero_field(presence, inputs, [address])[0, 0, 0].detach()
                        assert float(expected.abs().sum()) == 0.
                        torch.testing.assert_close(evidence[indices, b, e].reshape(-1, 2),
                            expected.expand(int(indices.sum()), 2), msg=str((configuration, text, word, lo, hi)))
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
    # Numeric learning is covered here; the verbal admission boundary is
    # retained by test_definition_integrity::test_word_concepts_exist_before_grounded_case_discovery.
    from test_grounded_xor import test_native_unseeded_xor
    test_native_unseeded_xor(tmp_path,pool,trial)


def test_training_admission_keeps_period_percept_without_lexical_identity(monkeypatch):
    from Corpus import admit_training_forms
    from Models import BaseModel
    from data import TheData
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    TheData.processLM({s: dict(text=['alpha.'] if s == 'train' else [], label=[])
                      for s in ('train', 'validation', 'test')})
    model = BaseModel.from_config('data/MM_corpus_orders.xml', data=TheData)[0]
    try:
        owner = model._concept_owner()
        percepts = model.perceptualSpace.percept_store
        admitted = nominated_forms(model, ['alpha.'])
        assert len(admitted) == 1 and admitted[0][0] == b'alpha'
        assert owner.interpret.word_for_form('alpha') == admitted[0][1]
        assert owner.interpret.word_for_form('.') is None
        assert b'.' in percepts.identity.words
        assert percepts.bytes_for(percepts.identity.word_rows[b'.']) == b'.'
        from Meronomy import percept_spans
        assert (5, 6) in percept_spans(b'alpha.')
    finally:
        model.End()


@pytest.mark.parametrize('surface', ['', ' ', '.', ' \u2014\u3002 ', b'\xe2\x80\x94'])
def test_eager_admission_never_looks_up_nonlexical_forms(monkeypatch, surface):
    from test_concept_capacity_policy import _cs, _property_ws, _D
    cs = _cs(32)
    calls = []
    lookup = cs.interpret.lookup_word
    def observed(parts, wholes, **kw):
        calls.append(kw.get('form'))
        return lookup(parts, wholes, **kw)
    monkeypatch.setattr(cs.interpret, 'lookup_word', observed)
    assert recurrent_interpretation(cs, [10], (), key=surface) is None
    if isinstance(surface, str):
        cs._autobind_property_concepts(torch.tensor([[10]]), torch.zeros(1, 1, _D),
            torch.tensor([[0]]), [[surface]], [[surface]], percept_where=None,
            percept_when=None, tile_spans=None, percept_store=None, ws=_property_ws())
    assert calls == []
    assert cs.interpret.word_for_form(surface) is None


def test_grammar_word_admission_stages_an_empty_bank_when_no_word_is_observed():
    from types import SimpleNamespace
    from Models import BasicModel
    raw = torch.tensor([[97, 98]])
    ids, spans = torch.tensor([[-1]]), torch.tensor([[[0, 2]]])
    staged = []
    model = SimpleNamespace(inputSpace=SimpleNamespace(_last_sentences=['ab']),
        perceptualSpace=SimpleNamespace(_forward_input={'native_indices':ids,'native_part_spans':spans}),
        wholeSpace=SimpleNamespace(), _staged_concepts_in=raw,
        _concept_owner=lambda: None, _aligned_serial_word_mode=lambda: False,
        _stage_mixing_grammar_leaf_mask=lambda: None,
        _stage_reading_grammar_references=lambda words, fi: staged.append(words))
    BasicModel._stage_reading_word_concepts(model)
    assert staged == [[]]
    assert model._reading_word_percepts is None
    assert model._reading_word_extents is None


def test_grammar_word_admission_keeps_nonlexical_percept_extent(monkeypatch):
    from types import SimpleNamespace
    from Models import BasicModel
    import Meronomy
    lookups = []
    owner = SimpleNamespace(interpret=SimpleNamespace(lookup_word=lambda *a, **kw: lookups.append(kw)))
    raw = torch.tensor([[46]])
    ids, spans = torch.tensor([[0]]), torch.tensor([[[0, 1]]])
    # Exercise the lexical guard independently of the current word-span
    # splitter: even a supplied grammar-word span cannot admit punctuation.
    monkeypatch.setattr(Meronomy, 'word_spans', lambda data: [(0, 1)])
    monkeypatch.setattr(Meronomy, 'percept_spans', lambda data: [(0, 1)])
    captured = []
    model = SimpleNamespace(inputSpace=SimpleNamespace(_last_sentences=['.']),
        perceptualSpace=SimpleNamespace(_forward_input={'native_indices':ids,'native_part_spans':spans}),
        wholeSpace=SimpleNamespace(inputShape=(1,), property_rows_for_bytes=lambda raw: (),
            concept_evidence_layout=lambda raw, width: (spans,None), subspace=SimpleNamespace(what=None)),
        _staged_concepts_in=raw, _concept_owner=lambda: owner,
        _aligned_serial_word_mode=lambda: False, _stage_mixing_grammar_leaf_mask=lambda: None,
        _stage_reading_grammar_references=lambda words, fi: captured.extend(words))
    BasicModel._stage_reading_word_concepts(model)
    assert not lookups and not captured
    torch.testing.assert_close(model._reading_word_extents, spans)


@pytest.mark.parametrize('surface, expected', [('', False), (b'', False), (' \t', False),
    ('\u2014\u3002', False), (' \u2014 ', False), (b'\xe2\x80\x94', False),
    ('alpha', True), ('alpha.', True), ('2', True), ('+', True)])
def test_lexical_form_predicate_uses_unicode_categories(surface, expected):
    from WordSurfaces import is_lexical_form
    assert is_lexical_form(surface) is expected
