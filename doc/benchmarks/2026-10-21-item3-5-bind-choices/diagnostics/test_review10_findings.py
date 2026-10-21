"""Isolate the reviewed ownership and root-role defects without a keep override."""
from dataclasses import replace

import pytest
import torch

from BindingAnswers import cost
from ClauseRow import Clause
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ThoughtReferences import open_slots, with_slots
from test_medium_term_memory import keep, mention, snapshot, unchanged
from test_reference_arrivals import named, world
from test_reference_choices import noun


def kept_arrival(resolver, site, selected):
    """Exercise commit after a candidate has already been selected."""
    pool = resolver.components.owner.mtm
    item = dict(site=site, site_occurrence=site, token=pool.token_of(site),
                selected=selected, candidates=(selected,))
    before = snapshot(resolver)
    resolver.commit(dict(items=[], arrivals=[item]), None, {},
                    document='review', position=4, training=False)
    unchanged(resolver, before)


def joined_world():
    resolver, store, atoms, words, model = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    site = store.occurrence_of(pending)
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']), position=1)
    first_key = store.occurrence_of(first)
    used, _ = keep(resolver, mention(atoms[0], mode='bind'),
                   (SituationFrame.from_row(store.row(first)),), position=2)
    use = store.occurrence_of(used)
    other, _ = keep(resolver, mention(atoms[1], reference=words['wrong']), position=3)
    other_key = store.occurrence_of(other)
    pool = resolver.components.owner.mtm
    original = pool.entries[first_key]
    pool.merge(site, first_key, when=pool.entries[other_key].when)
    return resolver, site, first_key, use, other_key, original


def test_later_pooled_arrival_moves_only_the_named_site():
    resolver, site, first, use, other, original = joined_world()
    pool = resolver.components.owner.mtm
    before = pool.entries[first]
    target = pool.entries[other]
    kept_arrival(resolver, site, other)
    assert pool.token_of(site) == other
    assert pool.token_of(first) == first and pool.token_of(use) == first
    assert pool.entries[first].kind == original.kind
    assert pool.entries[first].occurrences == original.occurrences
    assert pool.entries[first].count == before.count
    assert pool.entries[other].occurrences == (*target.occurrences, site)
    assert pool.entries[other].count == target.count + 1


def test_reselecting_the_current_owner_changes_no_pool_entry():
    resolver, site, first, use, other, original = joined_world()
    pool = resolver.components.owner.mtm
    before = dict(pool.entries)
    kept_arrival(resolver, site, use)
    assert pool.entries == before


@pytest.mark.parametrize('nested', (False, True))
def test_answer_judge_uses_root_roles_not_nested_local_indices(nested):
    resolver, store, atoms, words, model = world()
    antecedent, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    child = noun(atoms, words)
    if nested:
        child = Clause(ConceptualMeaning.from_description(atoms[0]),
                       refs=(('clause', 0), -1, -1), children=(child,))
    role = 0 if nested else 2
    meaning = with_slots(ConceptualMeaning.from_description(atoms[1]),
                         (('referent', role),), pair=(0., 0.))
    clause = Clause(meaning, refs=(-1, -1, ('clause', 0)), children=(child,))
    before = snapshot(resolver)
    entries = dict(resolver.components.owner.mtm.entries)
    plan = resolver.preview(clause, (SituationFrame.from_row(store.row(antecedent)),))
    assert plan['items'][0]['selected'] == store.occurrence_of(antecedent)
    judged = resolver.answer_meaning(plan, meaning)
    assert cost(model, judged, 'three') == (1 if nested else 0)
    assert bool(open_slots(judged)) == nested
    unchanged(resolver, before)
    assert resolver.components.owner.mtm.entries == entries
