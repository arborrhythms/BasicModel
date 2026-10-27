"""Scoped reader and reporting regressions found by item 3.0's full sweep."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch


def test_semantic_word_chain_does_not_redefine_a_word_form_order():
    from Layers import ConceptAllocator
    alloc = ConceptAllocator()
    left, right = alloc.new_concept(), alloc.new_concept()
    for word, atom in ((left, 17), (right, 23)):
        alloc.symbol_types.setdefault(word, set()).add('word')
        alloc.add(word, 'part', atom)
        alloc.reference_orders[word] = 0
    row = alloc.parts.add(left, right)
    assert ('sym', left) in alloc.refs(right, 'part')
    assert alloc.parts.find(left, right) == row
    assert alloc.settle(left) == alloc.settle(right) == 0
    assert alloc.order_of(right) == 0
    # Real concept definitions still enforce the inherited strict order check.
    parent = alloc.new_concept()
    alloc.reference_orders[parent] = 0
    alloc.parts.add(left, parent)
    with pytest.raises(ValueError, match='disagrees with its parts'):
        alloc.settle(parent)


def test_closing_mask_uses_resolved_journal_references_and_keeps_open_sites(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from Models import _open_reference_mask
    from ThoughtReferences import with_slots
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'wug', 'dax'))
    assert clause.meaning.role_refs == (None, None, None)
    assert not _open_reference_mask(clause.meaning, f.registry, clause).any()
    missing = replace(clause, refs=(-1, *clause.refs[1:]))
    expected = torch.zeros(3)
    expected[f.registry._slot_for_operand('argin1')] = 1
    torch.testing.assert_close(_open_reference_mask(missing.meaning, f.registry, missing), expected)
    marked = with_slots(replace(clause.meaning, bindings={
        '_forward_references': ((0, clause.refs[0]),)}), (('referent', 0),))
    torch.testing.assert_close(_open_reference_mask(marked, f.registry, replace(clause, meaning=marked)), expected)


def test_reconstruction_report_batches_within_constructed_streams_and_prints_every_row(monkeypatch):
    from Models import BasicModel
    model = BasicModel()
    rows = torch.arange(5, dtype=torch.float32).reshape(5, 1)
    model.inputSpace = SimpleNamespace(data=SimpleNamespace(test_input=rows, test_output=rows))
    model.conceptualSpace = SimpleNamespace(nStreams=2)
    visited, decoded = [], []
    monkeypatch.setattr(model, 'set_sigma', lambda value: None)
    def epoch(*, batchSize, split):
        assert split == 'test' and batchSize == 2
        for start in range(0, len(rows), batchSize):
            visited.extend(range(start, min(start+batchSize, len(rows))))
        return None, None, rows, rows
    monkeypatch.setattr(model, 'runEpoch', epoch)
    def decode(values, originals):
        decoded.extend(torch.as_tensor(originals).flatten().tolist())
        torch.testing.assert_close(values, rows)
        return ['x'] * len(values)
    monkeypatch.setattr(model, '_decode_reconstructed_inputs', decode)
    monkeypatch.setenv('BASIC_RECON_REPORT_MAX', '0')
    model._reconstructionReport()
    assert visited == list(range(5)) and decoded == rows.flatten().tolist()
    assert model.conceptualSpace.nStreams == 2


def test_noncanonical_forward_preserves_staged_symbol_slab_views(monkeypatch, eager_reading):
    from test_output_walk import _model
    from PrependedSymbols import word_view
    model = _model()
    model._tensor_peer_while_eager = True
    model._tensor_peer_while_enabled = False  # Exercise the existing fallback path.
    model._chart_compose_per_word = lambda: None
    pipeline = model._run_tensor_peer_word_pipeline
    observed = []
    def checked(*args, **kwargs):
        assert not model._tensor_peer_while_ready(args[0])
        names = ('concept_rows', 'concept_orders', 'object_rows', 'object_orders')
        before = {name: word_view(model.inputSpace, name).clone() for name in names}
        assert before['concept_rows'].ge(0).any()
        pointers = {name: word_view(model.inputSpace, name).data_ptr() for name in names}
        result = pipeline(*args, **kwargs)
        for name in names:
            torch.testing.assert_close(word_view(model.inputSpace, name), before[name], rtol=0, atol=0)
            assert word_view(model.inputSpace, name).data_ptr() == pointers[name]
        observed.append(True)
        return result
    monkeypatch.setattr(model, '_run_tensor_peer_word_pipeline', checked)
    try:
        with torch.no_grad():
            model(model.inputSpace.prepInput(['wug dax', 'dax wug']))
        assert observed == [True]
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_reconstruction_benchmark_respects_stream_capacity_and_keeps_batch_budget(tmp_path, monkeypatch):
    import recon_bench
    from configuration_fixtures import small_retained
    from Models import BasicModel
    observed = []
    epoch = BasicModel.runEpoch
    def bounded(self, *args, **kwargs):
        if kwargs.get('split') == 'test':
            n = len(self.inputSpace.getTestData()[0])
            cap = self._concept_owner().nStreams
            assert n > cap  # Reproduce the former oversized decode batch.
            assert kwargs['batchSize'] == cap
            assert kwargs['max_batches'] == 1
            observed.append((n, cap))
        return epoch(self, *args, **kwargs)
    monkeypatch.setattr(BasicModel, 'runEpoch', bounded)
    with small_retained('MM_ladder.xml') as config:
        record = recon_bench.run_config(config, epochs=0, seed=None,
            out_dir=str(tmp_path), max_batches=1)
    assert observed
    assert record.notes['decoded_rows'] <= observed[0][1]


def test_numerical_chain_reader_reads_both_references_without_writing_a_predicate():
    from part_fixtures import native_store, part
    from Spaces import ConceptualSpace
    from reasoning import TruthGroundedReasoner
    store, _ = native_store(width=6)
    left, right = torch.eye(6)[:2]
    edge = part(store, left, right, evidence=(.8, .2))
    before = store.parts.ref_kinds.clone(), store.parts.refs.clone(), len(store.parts)
    # The untyped working view intentionally occupies slots zero and one.
    torch.testing.assert_close(store.row(edge)['meaning'].roles[1], right)
    rows = list(ConceptualSpace._iter_relation_rows(store))
    assert len(rows) == 1 and rows[0][0] == edge[1]
    torch.testing.assert_close(rows[0][1], left, rtol=0, atol=0)
    torch.testing.assert_close(rows[0][2], right, rtol=0, atol=0)
    assert rows[0][3] == pytest.approx(.6)
    hit = TruthGroundedReasoner(store=store).query(left, right)
    assert hit['row'] == edge[1] and hit['match'] == pytest.approx(1.)
    torch.testing.assert_close(hit['np2'], right, rtol=0, atol=0)
    torch.testing.assert_close(store.parts.ref_kinds, before[0], rtol=0, atol=0)
    torch.testing.assert_close(store.parts.refs, before[1], rtol=0, atol=0)
    assert len(store.parts) == before[2]


def test_numerical_idea_query_reads_the_single_point_column():
    from Layers import LTM0
    from reasoning import TruthGroundedReasoner
    store = LTM0(6, capacity=3, percept_width=0)
    first, wanted = torch.eye(6)[:2]
    store.append_idea(first, trust=.3)
    row = store.append_idea(wanted, trust=.7)
    assert 'np1' not in store.row(row)
    hit = TruthGroundedReasoner(store=store).query(wanted)
    assert hit['row'] == row and hit['kind'] == 'idea' and hit['match'] == pytest.approx(1.)
    torch.testing.assert_close(hit['idea'], wanted, rtol=0, atol=0)
    assert hit['trust'] == pytest.approx(.7) and len(store) == 2
