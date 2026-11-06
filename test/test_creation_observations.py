"""Thinking §48 item 95: creation observations cannot be replaced by live reads."""
from dataclasses import replace

import pytest

from Meaning import ConceptualMeaning
from PresentationMemory import initial_observation, _initial_relation
from test_epoch_start import _supply_relation
from test_presentation_memory import begin, fixture, model_for, write


@pytest.mark.parametrize('memory', ['occurrence', 'relation'])
def test_missing_creation_observation_raises(memory):
    resolver, store, points = fixture()
    owner = resolver.components.owner
    begin(model_for(owner, store))
    if memory == 'occurrence':
        reference = write(store, points[0], 'missing-creation', surprise=.2)
        row = store.location(reference)
        del store._epoch_rows['initial_observations'][reference[-1]]
        store.access_count[row], store.surprise[row] = 7, .9
        with pytest.raises(ValueError, match=f'LTM0 occurrence {reference[-1]}.*creation observation'):
            initial_observation(store, row)
    else:
        row = store.parts.add(owner.new_concept(), owner.new_concept(), truth_plus=.6, surprise=.2)
        del store.parts.epoch_start['_initial_observations'][row]
        store.parts.access_count[row], store.parts.surprise[row] = 7, .9
        with pytest.raises(ValueError, match=f'LTM1 relation {row}.*creation observation'):
            _initial_relation(store.parts, row)


@pytest.mark.parametrize('supplied_before_start', [False, True])
def test_withdrawn_occurrence_later_referred_to_keeps_start_observation(supplied_before_start):
    resolver, store, points = fixture()
    model = model_for(resolver.components.owner, store)
    origin = store.ORIGIN_USER if supplied_before_start else store.ORIGIN_CONVERSATION
    dependency = write(store, points[0], 'withdrawn-start',
                       origin if supplied_before_start else None, surprise=.2)
    row = store.location(dependency)
    store.access_count[row], store.surprise[row] = 3, .4
    # The supply's snapshot was taken at admission. Reassert the intended setup
    # before the epoch so both setup paths begin with the same observation.
    if supplied_before_start:
        store.set_origin(row, origin, asserted=(1., 0.))
    begin(model)
    row = store.location(dependency)
    store.access_count[row], store.surprise[row] = 11, .9
    store.withdraw_origin(origin)
    assert store.location(dependency) is not None
    assert int(store.access_count[row]) == 11
    assert float(store.surprise[row]) == pytest.approx(.9)
    meaning = replace(ConceptualMeaning.from_description(points[1]), role_refs=(dependency, None, None))
    given = store.append_meaning(meaning, evidence=(.7, .1), document_key='later-supply')
    store.set_origin(given, store.ORIGIN_PROVISIONED, asserted=(.7, .1))
    for _ in range(2):
        begin(model)
        row = store.location(dependency)
        assert store.row(row)['evidence'] == (0., 0.)
        assert int(store.access_count[row]) == 3
        assert float(store.surprise[row]) == pytest.approx(.4)
        store.access_count[row], store.surprise[row] = 15, .8


def test_withdrawn_pre_epoch_relation_later_referred_to_keeps_start_observation():
    resolver, store, points = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    left, right = owner.new_concept(), owner.new_concept()
    child = _supply_relation(store, points[0], left, right, trust=.6)
    begin(model)
    expected = (int(store.parts.access_count[child]), float(store.parts.surprise[child]))
    store.parts.access_count[child], store.parts.surprise[child] = 11, .9
    store.withdraw_origin(store.ORIGIN_USER)
    assert store.parts.contains(child)
    given = write(store, points[1], 'later-relation-supply', store.ORIGIN_PROVISIONED)
    _supply_relation(store, points[1], given[-1], child)
    for _ in range(2):
        begin(model)
        assert store.parts.contains(child)
        assert float(store.parts.truth_plus[child]) == float(store.parts.truth_minus[child]) == 0.
        assert int(store.parts.access_count[child]) == expected[0]
        assert float(store.parts.surprise[child]) == pytest.approx(expected[1])
        store.parts.access_count[child], store.parts.surprise[child] = 15, .8


@pytest.mark.parametrize('memory', ['occurrence', 'relation'])
@pytest.mark.parametrize('release', ['withdrawal', 'compaction'])
def test_supply_closure_dependency_keeps_start_observation(memory, release):
    """A dependency outlives the revoked root without becoming its supply."""
    resolver, store, points = fixture()
    owner = resolver.components.owner
    model = model_for(owner, store)
    begin(model)
    if memory == 'occurrence':
        dependency = write(store, points[0], 'closure-history', surprise=.2)
        reference = dependency
        def observation():
            row = store.location(dependency)
            assert row is not None
            return store.access_count, store.surprise, row
    else:
        child = store.parts.add(owner.new_concept(), owner.new_concept(), truth_plus=.6, surprise=.2)
        reference = ('sym', child)
        def observation():
            assert store.parts.contains(child)
            return store.parts.access_count, store.parts.surprise, child
    meaning = replace(ConceptualMeaning.from_description(points[1]), role_refs=(reference, None, None))
    given = store.append_meaning(meaning, evidence=(.7, .1), document_key='closure-supply')
    store.set_origin(given, store.ORIGIN_USER, asserted=(.7, .1))
    supplied = store.occurrence_of(given)
    begin(model)
    counts, surprise, row = observation()
    expected = int(counts[row]), float(surprise[row])
    assert expected == pytest.approx((1, .2))
    counts[row], surprise[row] = 11, .9
    if release == 'withdrawal':
        store.withdraw_origin(store.ORIGIN_USER)
    else:
        removed = store.location(supplied)
        store.compact_rows([index for index in range(len(store)) if index != removed])
        assert store.location(supplied) is None
    counts, surprise, row = observation()
    assert int(counts[row]) == 11
    assert float(surprise[row]) == pytest.approx(.9)
    replacement = store.append_meaning(meaning, evidence=(.8, 0.), document_key='replacement-supply')
    store.set_origin(replacement, store.ORIGIN_PROVISIONED, asserted=(.8, 0.))
    for _ in range(2):
        begin(model)
        counts, surprise, row = observation()
        assert int(counts[row]) == expected[0]
        assert float(surprise[row]) == pytest.approx(expected[1])
        counts[row], surprise[row] = 15, .8


@pytest.mark.parametrize('memory', ['occurrence', 'relation'])
def test_current_checkpoint_missing_creation_observation_raises(memory, tmp_path):
    """A current envelope cannot silently repair a corrupted creation map."""
    import torch
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG, stateless=False)
    try:
        store = model.symbolSpace.ltm_store
        owner = model._concept_owner()
        begin(model)
        if memory == 'occurrence':
            point = owner.similarity_codebook.W[0].detach().clone()
            reference = write(store, point, 'missing-checkpoint-observation', surprise=.2)
            identity = reference[-1]
        else:
            identity = store.parts.add(owner.new_concept(), owner.new_concept(), truth_plus=.6, surprise=.2)
        path = tmp_path / 'missing-observation.pt'
        model.save_weights(str(path))
        saved = torch.load(path, weights_only=False)
        extras = saved['structural_extras']
        if memory == 'occurrence':
            for start in extras['epoch_rows'].values():
                if start is not None:
                    start.get('initial_observations', {}).pop(identity, None)
        else:
            for entry in extras['conceptual_spaces'].values():
                start = (entry.get('allocator') or {}).get('epoch_relations', (None,))[0]
                if start is not None:
                    start.get('_initial_observations', {}).pop(identity, None)
        torch.save(saved, path)
        tag = 'LTM0 occurrence' if memory == 'occurrence' else 'LTM1 relation'
        with pytest.raises(ValueError, match=f'{tag} {identity}.*creation observation'):
            model.load_weights(str(path), require_match=True)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_legacy_supplied_rows_import_saved_observation():
    """Older supply snapshots, not later live reads, define their observation."""
    from PresentationMemory import _rows
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG, stateless=False)
    try:
        store = model.symbolSpace.ltm_store
        point = model._concept_owner().similarity_codebook.W[0].detach().clone()
        begin(model)
        supplied = write(store, point, 'legacy-supply', store.ORIGIN_USER, surprise=.2)
        row = store.location(supplied)
        saved = _rows(store, [row])
        extras = model._collect_structural_extras()
        extras['version'] = 2
        for name, start in extras['epoch_rows'].items():
            if start is not None and supplied[-1] in start['columns']['address_keys'].tolist():
                extras['epoch_rows'][name] = dict(start['_base'], supplied_rows={supplied[-1]: saved})
        store.access_count[row], store.surprise[row] = 11, .9
        model._restore_structural_extras(extras)
        assert store._epoch_rows['initial_observations'][supplied[-1]] == pytest.approx((1, .2))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('memory', ['occurrence', 'relation'])
def test_legacy_checkpoint_imports_unsnapshotted_observation(memory, tmp_path):
    """The explicit old envelope imports its recorded live observation once."""
    import torch
    from test_ltm_consolidation import _make_model, _SERIAL_CONFIG
    model = _make_model(_SERIAL_CONFIG, stateless=False)
    try:
        store = model.symbolSpace.ltm_store
        owner = model._concept_owner()
        begin(model)
        if memory == 'occurrence':
            point = owner.similarity_codebook.W[0].detach().clone()
            reference = write(store, point, 'legacy-unsnapshotted', surprise=.2)
            identity, row = reference[-1], store.location(reference)
            store.access_count[row], store.surprise[row] = 7, .9
        else:
            identity = store.parts.add(owner.new_concept(), owner.new_concept(), truth_plus=.6, surprise=.2)
            store.parts.access_count[identity], store.parts.surprise[identity] = 7, .9
        path = tmp_path / 'legacy-unsnapshotted.pt'
        model.save_weights(str(path))
        saved = torch.load(path, weights_only=False)
        extras = saved['structural_extras']
        extras['version'] = 2
        for start in extras['epoch_rows'].values():
            if start is not None:
                start.pop('initial_observations', None)
        for entry in extras['conceptual_spaces'].values():
            start = (entry.get('allocator') or {}).get('epoch_relations', (None,))[0]
            if start is not None:
                start.pop('_initial_observations', None)
        torch.save(saved, path)
        assert model.load_weights(str(path), require_match=True)
        if memory == 'occurrence':
            row = store.location(reference)
            store.access_count[row], store.surprise[row] = 11, .4
            assert initial_observation(store, row) == pytest.approx((7, .9))
        else:
            store.parts.access_count[identity], store.parts.surprise[identity] = 11, .4
            held = _initial_relation(store.parts, identity)
            assert (int(held['access_count']), float(held['surprise'])) == pytest.approx((7, .9))
    finally:
        model.End()
        model.symbolSpace.soft_reset()
