"""Observe a native runtime packed reading; no config or policy overrides.

Earlier supplementary probes incorrectly supplied validation questions for
their runtime packed input. This observer uses the existing runtime protocol
of test_sentence_compose, retaining the unchanged MM_grammar_wording XML.
"""
import json
from pathlib import Path
import torch
from test_mm_xor import _fresh_model

HERE = Path(__file__).resolve().parent


def test_definition_share_of_native_predictor_reads(monkeypatch):
    from Layers import InterSentenceLayer
    reports = []
    original = InterSentenceLayer.expect_next_meaning

    def observe(self, b=0, **kwargs):
        before = self._inter_last_meaning[b]
        value = original(self, b, **kwargs)
        pending = self._inter_last_meaning[b]
        if pending is not None and pending is not before:
            store = self._ltm_store
            sources = tuple(getattr(pending, 'source_occurrences', ()))
            kinds = []
            for occurrence in sources:
                row = None if store is None else store._index_occurrences.get(occurrence)
                kinds.append(None if row is None else int(store.rel_type[row]))
            reports.append(dict(source_count=len(sources),
                resolved_count=sum(kind is not None for kind in kinds),
                definition_count=sum(kind == getattr(store, 'REL_DEF', -1) for kind in kinds),
                relation_kinds=kinds))
        return value

    monkeypatch.setattr(InterSentenceLayer, 'expect_next_meaning', observe)
    model, _, _ = _fresh_model(str(Path('data/MM_grammar_wording.xml')))
    texts = [['hello world', 'loving there', 'hello there']]
    try:
        # The documented eager word-loop fixture leaves model objectives,
        # chooser policies, capacities and input configurations untouched.
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        packed = model.inputSpace.prepPackedInput(texts)
        model.runBatch(train=False, batchNum=0, batchSize=1, split='runtime',
                       batch_override=(packed, torch.empty(1, 0)))
    finally:
        store = model.symbolSpace.ltm_store
        rows = sum(row['resolved_count'] for row in reports)
        definitions = sum(row['definition_count'] for row in reports)
        payload = dict(configuration='data/MM_grammar_wording.xml', seed=None,
            input=texts, new_estimates=reports, predictor_rows=rows,
            predictor_definitions=definitions,
            definition_share=None if not rows else definitions / rows,
            store_rows=0 if store is None else len(store),
            definition_rows=0 if store is None else int((store.rel_type[:len(store)] == store.REL_DEF).sum()))
        (HERE / 'native-definition-context.json').write_text(json.dumps(payload, indent=2) + '\n')
        model.End()
        model.symbolSpace.soft_reset()
