"""Verify MTM's fallback decisions while the configured model learns its corpus."""
from types import SimpleNamespace


def test_corpus_candidate_matches_follow_the_two_coverages(monkeypatch):
    import FamiliarTier
    from Memory import MediumTermMemory
    from attention_thinking_certificate import lesson
    from reading_fixtures import use_eager_reading

    use_eager_reading(monkeypatch)
    candidate, shared_part = MediumTermMemory.match_candidate, FamiliarTier.shared_part
    comparisons, refused, exact = 0, 0, 0
    scores = []

    def observe_part(*args, **kwargs):
        score = shared_part(*args, **kwargs)
        scores.append(score)
        return score

    def observe_candidate(pool, *args, **kwargs):
        nonlocal comparisons, refused, exact
        scores.clear()
        selected = candidate(pool, *args, **kwargs)
        if scores:
            comparisons += len(scores)
            best = max(scores)
            if best < pool.match_part or scores.count(best) > 1:
                assert selected is None
                refused += 1
            else:
                assert selected is not None
                # Unequal member proportions reached .9999976 on the
                # corpus. A fallback must cover both operands in full.
                assert best == 1.
        elif selected is not None:
            exact += 1
        return selected

    monkeypatch.setattr(FamiliarTier, 'shared_part', observe_part)
    monkeypatch.setattr(MediumTermMemory, 'match_candidate', observe_candidate)
    presentations = []

    def after_batch(model, *, label, **kwargs):
        presentations.append(label['texts'])

    lesson(SimpleNamespace(batch=after_batch, records=presentations))
    assert len(presentations) == 16
    assert comparisons > 45 and refused and exact, (comparisons, refused, exact)
