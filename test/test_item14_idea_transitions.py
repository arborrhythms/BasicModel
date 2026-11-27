"""Item 1.4, the sequence links (Alec, 2026-11-27; plan §9.2).

"MTM can hold the sequence links. Basically, I think they are M1 relations
over M0 ideas." Where a chain is kept: "Idea store would be fine." And "for
now, the idea store has a graph of transitions which is not yet mined."
With ``architecture.ideaTransitions`` true, each idea kept in the idea store
at a sentence's commit is linked to the idea kept just before it in the same
stream, by a relation of LTM1 over the two rows. There is one LTM1: the idea
store's relations are the concept taxonomy's own (Alec: "relations in LTM1
should not be a part of a different store"). A link is told apart from the
other relations by its mark, and is read as neither a part nor a kind.
Nothing counts it, reads it or promotes it.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import pytest
import torch

SCENES = ([3, 2, 0, 50], [5, 5, 0, 200], [4, 4, 0, 120])


def _model(tmp_path, monkeypatch, on):
    """The naming model of item 1.6, with ``ideaTransitions`` true when ``on``."""
    from test_item16_naming import naming_model
    from reading_fixtures import use_eager_reading
    import test_item17_echo as echo
    if on:
        built = echo.echo_model

        def configured(tmp_path, patch, *, architecture_overrides=None):
            return built(tmp_path, patch, architecture_overrides=dict(
                architecture_overrides or {}, ideaTransitions='true'))
        monkeypatch.setattr(echo, 'echo_model', configured)
    model = naming_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    return model


def _occurrence_pairs(parts):
    """Every relation of two occurrence references, as (earlier, later) names."""
    if not len(parts):
        return []
    O = parts.OCCURRENCE
    return sorted((left[1], right[1]) for left, right in parts.reference_pairs(parts.indices()).values()
                  if left[0] == O and right[0] == O)


def _present_three(model, streams=1):
    """Each scene presented alone in each stream, one runBatch each (with
    ``model.End()`` after it); the names of the rows each kept."""
    from test_item16_scene_field import present
    store = model.symbolSpace.ltm_store
    kept = []
    for scene in SCENES:
        before = set(store.row_names[:len(store)].tolist())
        present(model, [dict(text='', scene=[scene])] * streams, train=False)
        kept.append([name for name in store.row_names[:len(store)].tolist() if name not in before])
    return kept


def _concepts(owner):
    """Each concept's order and the references among its records, by id."""
    alloc = owner._concept_allocator
    ids = {owner.concept_id_at_row(row) for row in range(int(owner.nVectors))} - {None}
    return {cid: (alloc.order_of(cid), tuple(ref for _, ref in alloc.records(cid))) for cid in sorted(ids)}


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """The same three scenes, with the key on and with it off. With it on,
    every concept's order and records are read before and after every
    link, and what LTM1's readers of parthood give at each link's ends."""
    import Memory
    record = {}
    for on in (True, False):
        with pytest.MonkeyPatch.context() as monkeypatch:
            calls = []
            if on:
                adding = Memory.LTM1.add_transition

                def counted(parts, before, after):
                    concepts = _concepts(owner)
                    row = adding(parts, before, after)
                    O = Memory.OCCURRENCE
                    read = {name: (parts.parts((O, name)), parts.wholes((O, name))) for name in (before, after)}
                    calls.append(dict(link=(before, after), row=row, concepts=(concepts, _concepts(owner)),
                                      read=read, handle=(parts.parts(row), parts.wholes(row))))
                    return row
                monkeypatch.setattr(Memory.LTM1, 'add_transition', counted)
            model = _model(tmp_path_factory.mktemp("on" if on else "off"), monkeypatch, on)
            owner = model._concept_owner()
            try:
                store = model.symbolSpace.ltm_store
                kept = _present_three(model)
                record['on' if on else 'off'] = dict(
                    key=model.idea_transitions, kept=kept, pairs=_occurrence_pairs(store.parts),
                    relations=len(store.parts), marked=sorted(store.parts.transition_rows()),
                    taxonomy=len(owner._concept_allocator.parts), calls=calls,
                    one=store.parts is owner._concept_allocator.parts,
                    rows={row: pair for row, pair in store.parts.reference_pairs(store.parts.indices()).items()})
            finally:
                model.End()
    return record


def test_each_kept_idea_is_linked_to_the_one_kept_before_it(measured):
    """Three scenes, one after another, each its own runBatch: the idea
    store's relations are exactly first -> second and second -> third, as
    references to those rows; none to itself and none backwards."""
    on = measured['on']
    print(on)
    assert on['key']
    assert [len(rows) for rows in on['kept']] == [1, 1, 1]
    (first,), (second,), (third,) = on['kept']
    assert on['pairs'] == sorted([(first, second), (second, third)])
    assert on['relations'] == 2 and len(on['marked']) == 2
    assert all(a != b for a, b in on['pairs'])
    assert (second, first) not in on['pairs'] and (third, second) not in on['pairs']


def test_the_links_are_in_the_one_ltm1_marked_and_neither_parts_nor_kinds(measured):
    """There is one LTM1: the idea store's relations are the concept
    taxonomy's, and the links between kept ideas are written there. They
    are told apart from the other relations by their mark: the marked rows
    are exactly the links, and every other relation is as with the key off.
    A link is read as neither a part nor a kind: LTM1's readers of parthood
    give no idea as a part or a whole of another at either end of a link,
    nor anything for its handle; and no concept's order or records change
    when a link is written."""
    import Memory
    on, off = measured['on'], measured['off']
    O = Memory.OCCURRENCE
    links = {row: (left[1], right[1]) for row, (left, right) in on['rows'].items()
             if left[0] == right[0] == O}
    print(dict(one=on['one'], marked=on['marked'], links=links, taxonomy=(on['taxonomy'], off['taxonomy']),
               read=[call['read'] for call in on['calls']], handle=[call['handle'] for call in on['calls']]))
    assert on['one'] and off['one']
    assert len(on['calls']) == 2
    assert on['marked'] == sorted(links) == sorted(call['row'] for call in on['calls'])
    assert sorted(links.values()) == on['pairs']
    assert on['taxonomy'] - len(on['marked']) == off['taxonomy']
    for call in on['calls']:
        for name, (parts, wholes) in call['read'].items():
            for earlier, later in on['pairs']:
                assert not (later == name and (O, earlier) in parts)
                assert not (earlier == name and (O, later) in wholes)
        assert call['handle'] == ((), ())
        before, after = call['concepts']
        assert before == after
        assert not any(('rel', row) in refs for _, refs in after.values() for row in on['marked'])


def test_with_the_key_off_the_idea_store_relations_are_as_they_were(measured):
    off = measured['off']
    print(off)
    assert not off['key']
    assert [len(rows) for rows in off['kept']] == [1, 1, 1]
    assert off['pairs'] == [] and off['relations'] == 0 and off['marked'] == []


def test_streams_are_not_linked_to_each_other_and_a_hard_reset_begins_anew(tmp_path, monkeypatch):
    """Two streams read the same scenes side by side: each idea is linked
    only to the idea kept before it in its own stream. After a hard reset of
    stream 0 only, its next idea begins a new chain; stream 1's continues."""
    model = _model(tmp_path, monkeypatch, True)
    try:
        store = model.symbolSpace.ltm_store
        kept = _present_three(model, streams=2)
        pairs = _occurrence_pairs(store.parts)
        model.dispatch_per_row_reset([True, False])
        from test_item16_scene_field import present
        before = set(store.row_names[:len(store)].tolist())
        present(model, [dict(text='', scene=[SCENES[0]])] * 2, train=False)
        fourth = [name for name in store.row_names[:len(store)].tolist() if name not in before]
        later = sorted(set(_occurrence_pairs(store.parts)) - set(pairs))
        print(dict(kept=kept, pairs=pairs, fourth=fourth, later=later))
    finally:
        model.End()
    assert [len(rows) for rows in kept] == [2, 2, 2]
    # Rows are kept stream by stream, stream 0 first.
    chains = [[rows[b] for rows in kept] for b in (0, 1)]
    expected = sorted((chain[i], chain[i + 1]) for chain in chains for i in range(2))
    assert pairs == expected
    assert len(fourth) == 2
    assert later == [(chains[1][-1], fourth[1])]


def test_a_forgotten_idea_takes_its_links_with_it(tmp_path, monkeypatch):
    """The middle idea of three is forgotten: both its links go, the first
    and the third ideas stay, and they are not linked to each other. The
    next idea follows the third."""
    model = _model(tmp_path, monkeypatch, True)
    try:
        store = model.symbolSpace.ltm_store
        (first,), (second,), (third,) = _present_three(model)
        names = store.row_names[:len(store)].tolist()
        store.compact_rows([row for row, name in enumerate(names) if name != second])
        remaining = _occurrence_pairs(store.parts)
        protected = store.parts.protected_references()
        from test_item16_scene_field import present
        present(model, [dict(text='', scene=[SCENES[0]])], train=False)
        fourth = store.row_name(len(store) - 1)
        final = _occurrence_pairs(store.parts)
        print(dict(rows=(first, second, third, fourth), remaining=remaining, final=final,
                   protected=sorted(protected)))
    finally:
        model.End()
    assert remaining == [] and protected == set()
    assert final == [(third, fourth)]
