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
                            for row in result.materialize().reshape(len(args[1].where), -1).tolist())
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


def test_the_stm_head_names_the_operators_and_their_operands(tmp_path, monkeypatch, eager_reading):
    from Memory import OCCURRENCE, RELATION, SYMBOL
    from PartsOfSpeech import operation_name
    from SentenceStrata import Tree
    from test_item17_places import model_for
    from test_item18_expect import read

    model = model_for(tmp_path, monkeypatch, rules='surface')
    original = Tree.write_edges
    applications = []

    def observed(tree, store, concepts, where, when, language):
        result = original(tree, store, concepts, where, when, language)
        head = tree.trace.head

        def read_pair(ref):
            kind, name = ref
            if kind == OCCURRENCE:
                return store.referent_of(store.index_of_row(name))
            if kind == SYMBOL:
                return language.parts_of_speech.seats[name].name
            assert kind == RELATION
            return tuple(read_pair(child) for child in store.parts.references(name))

        # Compare a read from the single head with the operations the model
        # actually took, including their scope and the order of the operands.
        index = tree.trace.index
        expected = {}
        def sequence(values):
            first, *rest = values
            return (first, sequence(rest)) if rest else first
        for at, (kind, local, word) in enumerate(index.steps):
            if kind == 0:
                expected[at] = None if concepts[word] is None else ('sym', concepts[word])
            elif kind in (1, 2):
                rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
                name = operation_name(rules[local])
                expected[at] = name, sequence([expected[child] for child in index.operands[at]])
                applications.append(name)
        assert read_pair(head) == sequence([expected[root] for root in index.roots])
        owner = model._concept_owner()
        held = list(owner.mtm.read_occurrences(store, head))
        assert len(held) == len(concepts)
        assert Counter(cid for _, cid, _ in held) == Counter(concepts)
        for row, cid, address in held:
            if cid is not None:
                assert store.referent_of(row) == ('sym', cid)
                assert owner.concept_id_at_row(address) == cid
        return result

    try:
        nominate(model, ('This is Felix.',))
        monkeypatch.setattr(Tree, 'write_edges', observed)
        with torch.no_grad():
            for text in ('Felix.', 'This is Felix.', 'Felix Felix.'):
                read(model, text)
                model.End()
        assert 'surface' in applications
    finally:
        model.End()


def test_identical_words_carry_distinct_locations_between_looks(tmp_path, monkeypatch, eager_reading):
    from test_item18_expect import read
    model = _grammar_free_text_model(tmp_path, monkeypatch)
    try:
        nominate(model, ('Felix', 'Felix Felix.'))
        state = model.conceptualSpace.stm.model_state
        observed, requested = [], []
        observe, units = state.observe, model.request_read.units

        def selected(spans, index):
            observe(spans, index)
            if index[0] >= 0:
                observed.append(tuple(state.last_where[0].tolist()))

        def requested_at(x, state_values):
            if observed:
                wanted = state.last_where[0].clone()
                wanted[1] -= 1
                torch.testing.assert_close(state_values[0, :-4], model.where_registry.band(wanted).flatten())
                requested.append(tuple(state.last_where[0].tolist()))
            return units(x, state_values)

        monkeypatch.setattr(state, 'observe', selected)
        monkeypatch.setattr(model.request_read, 'units', requested_at)
        read(model, 'Felix Felix.')
        start, _ = model.where_registry.slices['text']
        words = {(start, start + 5), (start + 6, start + 11)}
        assert words.issubset(observed)
        assert words.intersection(requested)
    finally:
        model.End()


def test_reduced_head_copies_only_reachable_payload_and_original_stamp():
    from Memory import LTM0, STM0, OCCURRENCE, RELATION, SYMBOL
    from ReadingMemory import commit
    source = STM0(12, capacity=4, relation_capacity=32)
    source.append_idea(torch.ones(12))
    row = source.append_idea(torch.arange(12.), referent=('sym', 7), trust=.75)
    name = source.row_name(row)
    root = OCCURRENCE, name
    destination = LTM0(12, capacity=1, relation_capacity=8)
    kept, = commit([source], [root], destination)
    assert kept == (OCCURRENCE, destination.row_name(0))
    assert len(destination) == 1 and not len(destination.parts)
    assert torch.equal(destination.slots[0], source.slots[row])
    assert destination.referent_of(0) == ('sym', 7)

    # The same occurrence used twice must still have just one physical copy.
    pair = source.parts.add(root, root, truth_plus=.5, truth_minus=.25)
    application = source.parts.add((SYMBOL, 9), (RELATION, pair), truth_plus=1.)
    destination = LTM0(12, capacity=1, relation_capacity=8)
    kept, = commit([source], [(RELATION, application)], destination)
    assert kept[0] == RELATION and len(destination) == 1 and len(destination.parts) == 2
    operator, operands = destination.parts.references(kept[1])
    assert operator == (SYMBOL, 9)
    copied = OCCURRENCE, destination.row_name(0)
    assert destination.parts.references(operands[1]) == (copied, copied)
    assert destination.parts.truth_minus[operands[1]] == .25
    source.expire()
    assert torch.equal(destination.slots[0], torch.arange(12.))


def test_retained_occurrence_references_survive_stm_expiry_and_restore():
    import weakref
    from Memory import LTM0, STM0, OCCURRENCE
    from ReadingMemory import commit
    destination = LTM0(12, capacity=4, relation_capacity=16)
    previous = destination.append_idea(torch.ones(12), referent=('sym', 12))
    from SentenceStrata import IdeaStack
    source = IdeaStack(12, capacity=8, relation_capacity=16)
    source.address_domain = destination.address_domain
    source._external_memory = weakref.ref(destination)
    source.names_issued.fill_(10)
    durable_name = destination.row_name(previous)
    first = source.append_idea(torch.arange(12.), referent=('ltm', source.address_domain, durable_name))
    second = source.append_idea(-torch.arange(12.), referent=source.occurrence_of(first))
    leaf = OCCURRENCE, source.row_name(second)
    root, = commit([source], [leaf], destination)
    assert len(destination) == 3
    assert destination.referent_of(1) == ('ltm', destination.address_domain, durable_name)
    assert destination.referent_of(2) == destination.occurrence_of(1)
    source.expire()
    restored = LTM0(12, capacity=4, relation_capacity=16)
    restored.load_state_dict(destination.state_dict())
    row = restored.index_of_row(root[1])
    assert torch.equal(restored.slots[row], -torch.arange(12.))
    assert restored.referent_of(row) == restored.occurrence_of(1)


def test_bad_head_in_later_stream_cannot_partly_commit_batch():
    import pytest
    from Memory import LTM0, STM0, OCCURRENCE
    from ReadingMemory import commit
    source = STM0(12, capacity=2, relation_capacity=8)
    row = source.append_idea(torch.ones(12))
    target = LTM0(12, capacity=2, relation_capacity=8)
    with pytest.raises(ValueError, match="unavailable"):
        commit([source, source], [(OCCURRENCE, source.row_name(row)), (OCCURRENCE, 999)], target)
    assert len(target) == 0 and int(target.names_issued) == 0
