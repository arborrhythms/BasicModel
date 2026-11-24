"""Certificate 52 (sigma-pi §21, rule 60): the query on LTM matches the top of
STM's stack and pushes its top row there; and rule 65 as far as the key's
entry: an uncertain object fills no slot."""
from dataclasses import replace

import pytest
import torch
from architecture import WHERE_WIRES, WHEN_WIRES

# Every width of this world is one percept band (.where and .when): 8 -> 16.
_B = WHERE_WIRES + WHEN_WIRES


def world():
    """A conceptual space with four objects, an STM of one stream and an LTM
    of three rows whose entries are {a, b}, {c, d} and {a, c at one half}."""
    from Memory import ShortTermMemory, LTM0
    from clock_fixtures import for_stm
    from test_cs_sparse_weights import _cs
    from ClauseRow import attach_clause_index
    from EpisodicMemory import write
    from Meaning import ConceptualMeaning
    stm = ShortTermMemory(batch=1, concept_dim=_B)
    model = for_stm(stm)
    owner = _cs(nS=64, order=1, dim=_B)
    objects = [owner.new_concept(order=0) for _ in range(4)]
    owner.stm = stm
    store = LTM0(_B, capacity=8, percept_width=_B, episodic_concepts=4)
    attach_clause_index(model, store, owner)
    a, b, c, d = objects
    rows = []
    for subject, ids, values in ((a, [a, b], [1., 1.]), (c, [c, d], [1., 1.]), (a, [a, c], [1., .5])):
        meaning = replace(ConceptualMeaning.from_description(torch.ones(_B)), role_refs=(('sym', subject), None, None))
        row = store.append_meaning(meaning, trust=.8)
        write(store, row, (torch.tensor(ids + [-1] * (4 - len(ids))), torch.tensor(values + [0.] * (4 - len(values)))))
        rows.append(row)
    return SimpleWorld(owner, stm, store, objects, rows)


class SimpleWorld:
    def __init__(self, owner, stm, store, objects, rows):
        self.owner, self.stm, self.store, self.objects, self.rows = owner, stm, store, objects, rows

    def reader(self):
        from Queries import ThoughtLTMCapability
        return ThoughtLTMCapability(store=lambda: self.store, equal=lambda *_: 0., tau_id=.25, row=0)

    def idea(self, ids, *, question=False):
        """A thought's idea whose extension is ``ids`` at one, on top of the stack."""
        from Extension import ConceptSet, Node
        from Meaning import ConceptualMeaning
        from ThoughtChunks import put
        from ThoughtReferences import with_slots
        meaning = replace(ConceptualMeaning.from_description(torch.ones(_B)),
                          role_refs=(('sym', ids[0]), None, None),
                          extension=(Node(ConceptSet(torch.tensor(ids), torch.tensor([[1., 0.]] * len(ids)))),))
        if question:
            meaning = with_slots(meaning, (('referent', 2),), pair=(0., 0.))
        return put(self.stm, meaning, row=0, pin=True)

    def query(self, pattern):
        from QueryWork import QueryWorkBudget
        return self.reader().query(pattern, max_records=8, work=QueryWorkBudget(256))

    def top(self):
        """The row the top chunk of the stack points at, or None."""
        from Memory import STMChunks as chunks
        state = self.stm.state
        index = int(chunks.at(state)[0])
        if index < 0:
            return None
        kind, name = state[chunks.REFS][0, index, 0].tolist()
        if kind != chunks.OCCURRENCE:
            return 'thought'
        return self.store.location(('ltm', self.store.address_domain, int(name)))

    def depth(self):
        from Memory import STMChunks as chunks
        return int(self.stm.state[chunks.DEPTH][0])


def test_52a_a_question_on_top_pushes_the_row_that_answers_it_and_one_row_only():
    w = world()
    a, b, *_ = w.objects
    question = w.idea([a, b], question=True)
    before = w.depth()
    found = w.query(question)
    assert found['pushed'] and len(found['frames']) == 1
    assert w.depth() == before + 1
    assert w.top() == w.rows[0]


def test_52b_an_idea_that_is_no_question_pushes_the_row_most_like_it():
    w = world()
    _, _, c, d = w.objects
    idea = w.idea([c, d])
    found = w.query(idea)
    assert found['pushed']
    assert w.top() == w.rows[1]


def test_52c_a_second_query_does_not_push_the_row_the_first_pushed():
    w = world()
    a, b, *_ = w.objects
    question = w.idea([a, b], question=True)
    w.query(question)
    assert w.top() == w.rows[0]
    depth = w.depth()
    # The key is now the row just pushed; it is held on the stack and left
    # out, and the row most like it is pushed above it.
    found = w.query(question)
    pushed = [w.store.location(frame['occurrence']) for frame in found['frames']]
    assert w.rows[0] not in pushed
    assert w.top() != w.rows[0] and w.depth() == depth + (1 if found['pushed'] else 0)
    assert w.top() == w.rows[2]


def test_52d_nothing_on_the_stack_or_a_top_with_no_entry_pushes_nothing():
    from Understanding import ConceptualField
    w = world()
    field = ConceptualField(torch.tensor(w.objects[:2]), torch.tensor([[1., 0.], [1., 0.]])[:, None, None],
                            torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), None)
    assert w.top() is None
    found = w.query(field)
    assert not found['pushed'] and not found['frames'] and w.top() is None
    # A thought on top whose idea has no extension, and no grammar to give
    # one: no entry, so no key.
    from Meaning import ConceptualMeaning
    from ThoughtChunks import put
    put(w.stm, ConceptualMeaning.from_description(torch.ones(_B)), row=0, pin=True)
    depth = w.depth()
    found = w.query(field)
    assert not found['pushed'] and not found['frames']
    assert w.depth() == depth and w.top() == 'thought'


def test_65_an_uncertain_object_in_the_key_fills_no_slot_its_word_does():
    from EpisodicMemory import stm_key
    from Spaces import _concept_alloc_of
    w = world()
    a, b, c, d = w.objects
    alloc = _concept_alloc_of(w.owner)
    # ``b`` is a blank object, named by the one word ``d``.
    alloc.symbol_types.setdefault(b, set()).add('blank')
    w.owner.definitions.words = lambda cid: [d] if cid == b else []
    w.idea([a, b])
    key, = stm_key(w.store, w.stm, 0)
    ids = set(key.field.ids.tolist())
    assert b not in ids and {a, d} <= ids
