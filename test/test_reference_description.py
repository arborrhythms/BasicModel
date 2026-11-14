"""§52: ordinary descriptions retain their phrase, and exclusion's remaining uses."""
from dataclasses import replace

import pytest
import torch

from ClauseJournal import finish_clause
from ReferenceContext import SituationFrame
from chunk_fixtures import context as chunk_context
from reading_fixtures import record_reading
from test_clause_acceptance import SentenceFixture
from test_reference_close import fixture, close
from test_reference_review10 import observed_individual


def phrase(fixture, determiner, description, *, lexical=True):
    """A forced grammar reading, executed and closed by the native journal."""
    program = fixture.program(('lift', ('lower', 'a' if determiner == 'lower' else 'the',
                                        description), 'rolls'))
    if determiner != 'lower':
        mint = next(i for i, rule in enumerate(fixture.binary)
                    if rule.method_name == 'lower' and rule.determiner_mode == 'mint')
        bind = next(i for i, rule in enumerate(fixture.binary)
                    if rule.method_name == 'lower' and rule.determiner_mode == 'bind')
        actions = program.actions.clone()
        actions[(actions[:, 0] == 1) & (actions[:, 1] == mint), 1] = bind
        program = replace(program, actions=actions)
    orders = torch.ones(len(program.leaves), dtype=torch.long)
    orders[-1] = 0
    program = replace(program, close_references=True, leaf_orders=orders)
    if not lexical:
        program = replace(program, word_ids=torch.full_like(program.word_ids, -1))
    return finish_clause(fixture.language, record_reading(fixture.language, program),
                         registry=fixture.registry)


def close_sentence(resolver, clause):
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(kept,
                                      written_rows=written)
    resolver.commit(plan, row, written, document='description', position=0, training=False)
    frame = replace(SituationFrame.from_row(resolver.store.row(row)),
                    symbol_ids=written.get(('symbols', id(kept)), ()),
                    activations=written.get(('activations', id(kept)), ()))
    return frame, plan


@pytest.mark.parametrize('head_role', (0, 2), ids=('headless', 'noun-head'))
def test_ordinary_description_keeps_its_phrase_and_scores_by_containment(monkeypatch, head_role):
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    # Force only the parse's head annotation. Both use the same native union;
    # this is a mechanism certificate, not a claim about learned parsing.
    f.binary = tuple(rule._replace(head_role=head_role) if rule.method_name == 'union'
                     else rule for rule in f.binary)
    f.language._compose_binary_rules = f.binary
    for word in ('a', 'red', 'ball', 'blue', 'the', 'rolls'):
        f.noun(word)
    red = f.registry._payload(('sym', f.noun('red')))
    ball = f.registry._payload(('sym', f.noun('ball')))
    blue = f.registry._payload(('sym', f.noun('blue')))
    sentence = phrase(f, 'lower', ('union', 'red', 'ball'))
    frame, plan = close_sentence(resolver, sentence)
    item, = plan['items']
    assert len(item['leaves']) == 2
    assert item['node'].subject_word_id == f.words['ball' if head_role else 'red'][0]
    token = f.cs.mtm.token_of(item['reference'])
    assert token is not None
    expected = torch.maximum(red, ball)
    torch.testing.assert_close(f.cs.mtm.value_of(token), expected, rtol=0, atol=0)
    assert f.cs.mtm.reading_of(token) not in (f.words['red'][0], f.words['ball'][0])
    frames = (frame,)
    records = resolver.individuals(resolver.accessible(chunk_context(resolver, frames)))
    assert token in records
    predicate = f.registry._payload(('sym', f.noun('rolls')))
    cue = torch.zeros_like(red)
    alternatives, _, _ = resolver.score(torch.maximum(blue, ball), predicate, cue, 0, records)
    assert token not in alternatives
    contained, _, _ = resolver.score(ball, predicate, cue, 0, records)
    assert token in contained
    definite = resolver.preview(phrase(f, 'definite', ('union', 'blue', 'ball')),
                                chunk_context(resolver, frames))
    blue_item, = definite['items']
    assert blue_item['mode'] == 'bind'
    torch.testing.assert_close(blue_item['node'].point, torch.maximum(blue, ball), rtol=0, atol=0)
    assert token not in blue_item['candidates']


def test_one_leaf_close_retains_its_word_reading(monkeypatch):
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    _, plan = close_sentence(resolver, phrase(f, 'lower', 'ball'))
    item, = plan['items']
    token = f.cs.mtm.token_of(item['reference'])
    assert len(item['leaves']) == 1
    assert f.cs.mtm.reading_of(token) == f.words['ball'][0]


def test_description_promotion_does_not_define_its_derived_reading(monkeypatch):
    from FamiliarTier import promote_identity
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    # A non-lexical field has a native derived kind but no word reading.
    _, plan = close_sentence(resolver, phrase(f, 'lower', ('union', 'red', 'ball'), lexical=False))
    item, = plan['items']
    token = f.cs.mtm.token_of(item['reference'])
    reading = f.cs.mtm.reading_of(token)
    assert reading is None  # A description references its own idea, never its derived kind.
    assert 'word' not in f.cs._concept_allocator.symbol_types.get(reading, ())
    before = dict(f.cs._concept_allocator.named_identities)
    identity = promote_identity(f.cs, token)
    assert f.cs.mtm.entries[token].symbol == identity
    assert f.cs._concept_allocator.named_identities == before
    assert all(identity not in f.cs.definitions.objects(word) for word, kinds in f.cs._concept_allocator.symbol_types.items() if 'word' in kinds)


def test_asserted_constraint_still_excludes_a_contained_candidate():
    resolver, store, atoms = fixture()
    frame, _, plan = close(resolver, atoms[0] + atoms[1], atoms[6], 'mint', (),
                           document='constraint', sentence=0)
    token = plan['items'][0]['reference']
    for index, support in enumerate((atoms[1], atoms[2])):
        observed_individual(resolver, support, index)
    records = resolver.individuals(resolver.accessible(chunk_context(resolver, (frame,))))
    # Own content is a part of the antecedent. Only the separately asserted
    # coordinate conflicts with witnessed evidence on its other coordinate.
    args = (atoms[0], atoms[6], torch.zeros_like(atoms[0]), 0, records)
    allowed, _, _ = resolver.score(*args)
    assert token in allowed
    assert not resolver.exclusion_query()(atoms[0], atoms[0] + atoms[1])
    assert resolver.exclusion_query()(atoms[2], atoms[0] + atoms[1])
    excluded, _, _ = resolver.score(*args, asserted=(atoms[2],))
    assert token not in excluded


def test_witnessed_exclusion_still_splits_grouped_uses():
    resolver, store, atoms = fixture()
    first, _, plan = close(resolver, atoms[0], atoms[6], 'mint', (),
                                 document='group', sentence=0)
    token = plan['items'][0]['reference']
    frames = [first]
    for sentence, value in enumerate((atoms[0] + atoms[1], atoms[0] + atoms[2]), 1):
        # These explicit identity joins are independent of content association.
        # Grouping must still interpret the incompatible retained observations.
        frame, _, other = close(resolver, value, atoms[6], 'mint', (),
                             document='group', sentence=sentence)
        resolver.components.owner.mtm.merge(other['items'][0]['reference'], token, when=sentence)
        frames.append(frame)
    records = resolver.accessible(chunk_context(resolver, frames))
    assert sorted(tuple(record['mention_value'].tolist()) for _, record in records.values()) == sorted(
        tuple(value.tolist()) for value in (atoms[0], atoms[0] + atoms[1], atoms[0] + atoms[2]))
    assert len(resolver.individuals(records)) == 1
    for index, support in enumerate((atoms[1], atoms[2])):
        observed_individual(resolver, support, index)
    assert resolver.exclusion_query()(atoms[0] + atoms[1], atoms[0] + atoms[2])
    assert len(resolver.individuals(records)) == 2
