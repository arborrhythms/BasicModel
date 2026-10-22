"""A closing bind's uncovered description reaches R and chooser credit."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import Clause
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ReferenceReconstruction import cost
from test_medium_term_memory import keep, mention, snapshot, unchanged
from test_reference_arrivals import world, named
from test_reference_choices import noun


def description(atoms, words, point):
    node = noun(atoms, words)
    return replace(node, point=point, meaning=ConceptualMeaning.from_description(point),
        composition=replace(node.composition, point=point),
        reference_request=dict(node.reference_request, content=point))


def sentence_cost(resolver, plans, *, active=None, enabled=True, weight=.1):
    """Exercise the production registration with unrelated objectives zero."""
    from Models import BasicModel
    root = resolver.policy.new_zeros(len(plans), 12)
    active = torch.ones(len(plans), dtype=torch.bool) if active is None else active
    record = SimpleNamespace(primed=SimpleNamespace(meaning_start=None))
    observation = dict(reference_plans=plans, entries=(None,)*len(plans), meanings=(None,)*len(plans))
    model = SimpleNamespace(reconstruct_in_loop=True, _sentence_reconstruction=enabled,
        reconstruction_reference_weight=weight, loss=SimpleNamespace(reconstruction_scale=7.),
        inter_loss_weight=0., symbolSpace=SimpleNamespace(expectation=None),
        inputSpace=SimpleNamespace(_reconstruction_sentence_available=active[:, None]),
        _concept_owner=lambda: resolver.components.owner,
        _publish_sentence_scratch=lambda state: None,
        _trial_understanding=lambda *args: record,
        _tensor_pushed_ideas=root[:, None],
        _sentence_observation=lambda *args: observation,
        _reconstruct_trial=lambda *args: (root[:, None], root[:, 0], root[:, 0],
                                         active, root[:, 0]))
    lang = [None]*12
    lang[6], lang[10], lang[11] = root[:, None], root.repeat(1, 3)[:, None], active[:, None]
    BasicModel._sentence_path_cost(model, (None, lang, None), 0, active)
    return model._sentence_cost_registry


def test_bind_departure_changes_r_keep_and_the_resolvers_score_function():
    from SentenceCredit import components, comparison, score_function
    resolver, store, a, words, _ = world()
    first, _ = keep(resolver, mention(a[0], reference=words['three']))
    frame = SituationFrame.from_row(store.row(first))
    node = description(a, words, a[0]+a[1])
    with torch.no_grad():
        resolver.policy.zero_()
    before = snapshot(resolver)
    greedy = resolver.preview(node, (frame,))
    explore = resolver.preview(node, (frame,), departure=0)
    assert greedy['items'][0]['own_object']
    selected = explore['items'][0]
    assert not selected['own_object'] and selected['selected'] in selected['candidates']
    assert explore['decisions'][0]['departed']
    registries = [sentence_cost(resolver, [plan]) for plan in (greedy, explore)]
    parts = torch.stack([components(registry, None, torch.zeros(1)) for registry in registries], 1)
    torch.testing.assert_close(parts, torch.tensor([[[0., 0., 0.], [.05, 0., 0.]]]))
    audit = comparison(parts, torch.tensor([True]))
    assert audit['keep_decision'] == ['greedy'] and audit['deciding'] == ['reconstruction']
    decision = explore['decisions'][0]
    probability = decision['probabilities'][decision['selected']].reshape(1, 1)
    loss, _ = score_function(probability, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), audit['costs'])
    loss.sum().backward()
    assert resolver.policy.grad is not None and resolver.policy.grad.abs().sum() > 0
    unchanged(resolver, before)


def test_same_close_alias_charges_the_bound_mention_and_not_the_mint():
    resolver, _, a, words, _ = world()
    first = mention(a[0], reference=words['three'])
    second = description(a, words, a[0]+a[1])
    clause = Clause(ConceptualMeaning.from_description(a[0]+a[1]), point=a[0]+a[1],
        children=(first, second), refs=(('clause', 0), -1, ('clause', 1)))
    before = snapshot(resolver)
    plan = resolver.preview(clause, ())
    assert plan['items'][0]['mode'] == 'mint'
    assert plan['items'][1]['selected'] == ('new', 0) and not plan['items'][1]['own_object']
    assert 'value' not in plan['individuals'][('new', 0)]
    torch.testing.assert_close(cost(resolver, [plan], a[:1]), torch.tensor([.5]))
    unchanged(resolver, before)


@pytest.mark.parametrize('weight', (0., .1, .35))
def test_registration_sums_sites_and_includes_zero_cost_active_rows(weight):
    resolver, store, a, words, _ = world()
    first, _ = keep(resolver, mention(a[0], reference=words['three']))
    plan = resolver.preview(description(a, words, a[0]+a[1]), (SituationFrame.from_row(store.row(first)),))
    assert not plan['items'][0]['own_object']
    twice = dict(plan, items=plan['items']*2)
    empty = dict(plan, items=[])
    registry = sentence_cost(resolver, [twice, plan, empty, twice],
        active=torch.tensor([True, True, True, False]), weight=weight)
    torch.testing.assert_close(registry.total(objective='reconstruction'),
        torch.tensor([weight, weight/2, 0., 0.]))
    assert registry._terms['reconstruction.reference']['count'] == 1
    disabled = sentence_cost(resolver, [twice], enabled=False)
    assert 'reconstruction.reference' not in disabled._terms


@pytest.mark.parametrize('ending', ('own', 'mint', 'gap', 'empty', 'arrival', 'covered'))
def test_unpriced_endings_and_covering_candidates_have_zero_reference_cost(ending):
    resolver, store, a, words, _ = world()
    first, _ = keep(resolver, mention(a[0], reference=words['three']))
    frame = SituationFrame.from_row(store.row(first))
    if ending == 'own':
        plan = resolver.preview(description(a, words, a[1]), ())
    elif ending == 'mint':
        plan = resolver.preview(mention(a[1], reference=words['wrong']), (frame,))
    elif ending == 'gap':
        plan = resolver.preview(mention(a[1], mode='elision'), (frame,))
    elif ending == 'empty':
        plan = resolver.preview(named(a[0], words['x']), (frame,), departure=0)
        assert not plan['items'][0]['own_object']
    else:
        plan = resolver.preview(description(a, words, a[0]), (frame,))
        assert not plan['items'][0]['own_object']
        if ending == 'arrival':
            # An arrival is outside the closing-item sum, even with uncovered content.
            plan = dict(plan, arrivals=[dict(plan['items'][0], content=a[1])], items=[])
    before = snapshot(resolver)
    torch.testing.assert_close(cost(resolver, [plan], a[:1]), torch.zeros(1))
    unchanged(resolver, before)


def test_missing_candidate_coordinates_are_uncovered_description():
    resolver, _, a, words, _ = world()
    node = description(a, words, a[0]+a[-1])
    plan = dict(items=[dict(node=node, selected=('new', 0), mode='bind', content=node.point)],
                individuals={('new', 0): dict(content=a[0, :1])})
    torch.testing.assert_close(cost(resolver, [plan], a[:1]), torch.tensor([.5]))
