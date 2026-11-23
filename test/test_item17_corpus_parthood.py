"""Exercise the candidate reader's parthood decision on actual corpus presentations."""
import json
from pathlib import Path


def test_corpus_candidate_matches_follow_the_two_coverages(tmp_path, monkeypatch):
    import FamiliarTier
    from Memory import MediumTermMemory
    from reading_fixtures import use_eager_reading
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_item18_expect import read

    use_eager_reading(monkeypatch)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=128, conceptual_overrides={'attentionPromotion': True},
        training_overrides={'reconstructionPlacement': 'eager'})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    candidate, shared_part = MediumTermMemory.match_candidate, FamiliarTier.shared_part
    comparisons, refused = 0, 0
    scores = []

    def observe_part(*args, **kwargs):
        result = shared_part(*args, **kwargs)
        scores.append(result)
        return result

    def observe_candidate(pool, *args, **kwargs):
        nonlocal comparisons, refused
        scores.clear()
        result = candidate(pool, *args, **kwargs)
        if scores:
            comparisons += len(scores)
            best = max(scores)
            if best < pool.match_part or scores.count(best) > 1:
                assert result is None
                refused += 1
            else:
                assert result is not None
        return result

    monkeypatch.setattr(FamiliarTier, 'shared_part', observe_part)
    monkeypatch.setattr(MediumTermMemory, 'match_candidate', observe_candidate)
    curriculum = json.loads((Path(__file__).resolve().parents[1] /
                             'data/corpus-lessons.json').read_text())
    texts = dict.fromkeys(text for document in curriculum['stages'][0]['documents']
                          for record in document['records'] for text in record['sentences'])
    try:
        optimizer = model.getOptimizer()
        for text in texts:
            read(model, text, training=True, optimizer=optimizer)
            model.End()
        assert comparisons > 45 and refused > 0, (comparisons, refused)
    finally:
        model.End()
