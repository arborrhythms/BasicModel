"""Read through the model. Observe the live STM chain, its commit and the returned bytes."""
from collections import Counter

import torch

from test_item17_one_reconstruction import _grammar_free_text_model
from test_item17_places import nominate


def test_looks_extend_one_head_and_commit_it_once(tmp_path, monkeypatch, eager_reading):
    from Memory import STM1, OCCURRENCE, RELATION
    import ReadingMemory
    import SentenceField

    model = _grammar_free_text_model(tmp_path, monkeypatch)
    try:
        nominate(model, ('This is Felix.', 'What is this?', 'Felix Felix.'))
        durable = model.symbolSpace.ltm_store
        owner = model._concept_owner()
        append, commit = STM1.append_occurrence, ReadingMemory.commit
        reconstruct = SentenceField.reconstruction
        arriving, retained, surfaces = {}, [], []

        def appended(parts, head, occurrence):
            result = append(parts, head, occurrence)
            prior = arriving.get(parts, ())
            assert tuple(parts.leaves(result)) == (*prior, (OCCURRENCE, occurrence))
            assert len(parts) == len(prior)
            arriving[parts] = tuple(parts.leaves(result))
            return result

        def committed(stores, heads, target):
            before = len(target), len(target.parts)
            result = commit(stores, heads, target)
            counts = []
            for store, saved, source in zip(stores, result, heads, strict=True):
                current = tuple(store.parts.leaves(source))
                assert current == arriving.pop(store.parts, ())
                leaves = tuple(target.parts.leaves(saved))
                assert len(leaves) == len(set(leaves)) == len(current)
                names = []
                for local, remote in zip(current, leaves, strict=True):
                    src, dst = store.index_of_row(local[1]), target.index_of_row(remote[1])
                    assert store.referent_of(src) == target.referent_of(dst)
                    torch.testing.assert_close(store.episodic_values[src], target.episodic_values[dst])
                    assert torch.equal(store.episodic_concepts[src], target.episodic_concepts[dst])
                    assert (target.episodic_concepts[dst] >= 0).any()
                    names.append(owner.interpret.lexical_form(target.referent_of(dst)[1]))
                cursor = saved
                while cursor is not None and cursor[0] == RELATION:
                    first, cursor = target.parts.references(cursor[1])
                    assert first[0] == OCCURRENCE
                retained.append((saved, leaves, names))
                counts.append(len(current))
            assert len(target) - before[0] == sum(counts)
            assert len(target.parts) - before[1] == sum(max(0, n - 1) for n in counts)
            return result

        def reconstructed(*args, **kwargs):
            result = reconstruct(*args, **kwargs)
            surfaces.extend(bytes(row).rstrip(b'\0').decode('latin1')
                            for row in result[1].materialize().reshape(len(args[1].where), -1).tolist())
            return result

        monkeypatch.setattr(STM1, 'append_occurrence', appended)
        monkeypatch.setattr(ReadingMemory, 'commit', committed)
        monkeypatch.setattr(SentenceField, 'reconstruction', reconstructed)
        for batch, words in ((['This is Felix.', 'Felix'], [('This', 'is', 'Felix', '.'), ('Felix',)]),
                             (['What is this?', 'Felix Felix.'], [('What', 'is', 'this', '?'), ('Felix', 'Felix', '.')]),
                             (['is is is is is is is is', '   '], [('is',) * 8, ()])):
            with torch.no_grad():
                model.runBatch(train=False, optimizer=model.getOptimizer(), batchSize=len(batch),
                    batch_override=(model.inputSpace.prepInput(batch), torch.empty(len(batch), 0)))
            assert surfaces[-len(batch):] == [text.strip() for text in batch]
            for (_, _, names), wanted in zip(retained[-len(batch):], words, strict=True):
                assert Counter(names) == Counter(wanted)
            model.End()
        for saved, leaves, names in retained:
            assert tuple(durable.parts.leaves(saved)) == leaves
            print('durable head', saved, 'look order', names)
    finally:
        model.End()
