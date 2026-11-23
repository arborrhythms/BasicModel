"""The formula's word values and its end-of-read object evaluation (§55/68)."""
from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest
from nomination_fixtures import recurrent_word


def test_2_first_document_matches_recorded_object_readings(monkeypatch):
    import gzip
    import hashlib
    import json
    from pathlib import Path
    from comprehension_replay import replay
    path = Path(__file__).parent / 'fixtures/item18-comprehension-first-document.pt.gz'
    manifest = json.loads(path.with_name('item18-comprehension-first-document.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
    with gzip.open(path, 'rb') as source:
        fixture = torch.load(source, weights_only=False, map_location='cpu')
    assert fixture['source_snapshot'] == '7768411acaecf5d2a9ad58a39f2f0041c79f43cdc6d94fc1b79176bec5bf7fb6'
    records = fixture['records']
    assert len(records) == 7 and {r['sentence'] for r in records} == {1, 2, 3, 4}
    for record in records:
        evaluation, forest = replay(record, monkeypatch)
        expected = record['expected']
        site = (record['sentence'], record['path'])
        torch.testing.assert_close(evaluation.program.actions, expected['actions'], rtol=0, atol=0)
        torch.testing.assert_close(evaluation.ids, expected['objects'], rtol=0, atol=0)
        torch.testing.assert_close(evaluation.native_ids, expected['native_ids'], rtol=0, atol=0)
        for at, (kind, local, _) in enumerate(expected['actions'].tolist()):
            if kind not in (1, 2):
                continue
            before, after = expected['frames'][at], evaluation.frames[at]
            rules = (evaluation.language._compose_binary_rules if kind == 1
                     else evaluation.language._compose_unary_rules)
            torch.testing.assert_close(after, before, rtol=0, atol=1e-6,
                msg=lambda message: f'{site}, action {at}, {rules[local].method_name}: {message}\n'
                    f'old operands={before[:2].tolist()}\nnew operands={after[:2].tolist()}')
        clause = evaluation.result.clause
        torch.testing.assert_close(clause.point, expected['point'], rtol=0, atol=1e-6)
        torch.testing.assert_close(clause.meaning.roles, expected['roles'], rtol=0, atol=1e-6)
        assert clause.relation == expected['relation']
        def fields_match(actual, recorded):
            assert len(actual) == len(recorded)
            for node, old in zip(actual, recorded):
                torch.testing.assert_close(node.field.ids, old['ids'], rtol=0, atol=0)
                torch.testing.assert_close(node.field.values, old['values'], rtol=0, atol=1e-6)
                fields_match(node.children, old['children'])
        fields_match(forest, expected['extension'])


@pytest.mark.parametrize('reader', ['roles', 'backward'])
def test_comprehension_later_action_preserves_operand_views_and_gradients(reader):
    from Understanding import Reading
    from Comprehension import Evaluation
    from SentenceCompose import saved_sentence_values
    points = torch.tensor([[.2, .4], [.3, .1], [.1, .2]], requires_grad=True)
    actions = torch.tensor([[0, -1, 0], [0, -1, 1], [1, 0, -1],
                            [0, -1, 2], [1, 0, -1]])
    program = Reading(rows=torch.full((3,), -1), word_rows=torch.full((3,), -1),
        activations=torch.ones(3), leaves=points, actions=actions,
        targets=torch.zeros(5, dtype=torch.long), end_state=torch.zeros(3, 2),
        concept_ids=torch.full((3,), -1), word_ids=torch.full((3,), -1),
        leaf_orders=torch.zeros(3, dtype=torch.long), interpret_forced=False,
        operation_values=torch.zeros(5, 3, 2))
    model = SimpleNamespace(_concept_owner=lambda: SimpleNamespace(),
        languageSpace=SimpleNamespace(),
        symbolSpace=SimpleNamespace(ltm_store=SimpleNamespace(when_start=None)))
    rule = SimpleNamespace(method_name='sum', head_role=1)
    operator = SimpleNamespace(content_operand=lambda value: value,
        compose_from_grammar_context=lambda operands, context: operands[0]+operands[1])
    with saved_sentence_values():
        evaluation = Evaluation(model, program, 0)
        evaluation.leaf(0); evaluation.leaf(1)
        evaluation.operation(2, 1, 0, -1, rule, operator)
        left, right = evaluation.frames[2, :2].unbind()
        loss = left.square().sum()+right.square().sum()
        evaluation.leaf(2)
        evaluation.operation(4, 1, 0, -1, rule, operator)
        if reader == 'roles':
            torch.testing.assert_close(torch.stack((left, right)), points[:2], rtol=0, atol=0)
        loss.backward()
    expected = torch.cat((2*points[:2].detach(), torch.zeros_like(points[2:])))
    torch.testing.assert_close(points.grad, expected, rtol=0, atol=0)


def test_thought_trial_restores_its_native_store_before_applying_activity():
    from clock_fixtures import for_stm
    from Memory import ShortTermMemory
    from Meaning import ConceptualMeaning
    from ThoughtStrata import bound_to_sentence, write
    from WalkTrials import ThoughtTrialState
    from MemoryActivity import accessed, trial_activity
    stm = ShortTermMemory(batch=1, capacity=8, concept_dim=4)
    model = for_stm(stm)
    bound_to_sentence(stm._thought_stores, words=8, operators=40)
    store = stm._thought_stores[0][0]
    leaf = ConceptualMeaning.from_description(torch.ones(4))
    write(store, leaf, key=0)
    smaller = ThoughtTrialState(model)
    larger = replace(leaf, constituents=(leaf,) * 33)
    @trial_activity()
    def trial():
        view = write(store, larger, key=0)
        accessed(store, [len(store)-1])
        accessed(store.parts, [view.root[1]])
        return ThoughtTrialState(model), view.root[1], len(store)-1
    kept, relation, row = trial()
    smaller.restore()
    assert store.capacity == 32
    kept.restore()
    assert len(store) == 34
    assert float(store.activity[row]) == pytest.approx(.6)
    assert float(store.parts.activity[relation]) == pytest.approx(.6)


def test_1_word_deposit_keeps_its_value_row_and_order(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from Memory import ShortTermMemory, STMChunks
    import ChunkJournal
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._tensor_peer_while_eager = True
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['hi no', 'we go'])
    incoming, observed = [], []
    push = ShortTermMemory.functional_push_step_masked
    journal = ChunkJournal.push

    def deposit(*args, **kwargs):
        import inspect
        from Interpret import activate_code
        caller = inspect.currentframe().f_back.f_locals
        index = int(caller['index'])
        atoms = model.inputSpace._ar_word_concept_atoms[:, index]
        width = model._concept_owner().similarity_codebook.mereology.percept_event_width
        expected = activate_code(atoms, caller['activation'].reshape(len(atoms)), width,
                                 evidence=caller['leaf_evidence'])
        start = model.symbolSpace.ltm_store.when_start
        if start is not None:
            low = max(0, start-4)
            expected[:, low:start+4] = caller['symbolic_event'][:, 0, low:start+4]
        active = caller['commit'].reshape(-1)
        torch.testing.assert_close(args[1][active], expected[active], rtol=0, atol=0)
        state = push(*args, **kwargs)
        incoming.append(tuple(value.clone() for value in STMChunks.stack_view(state)[:6]))
        return state

    def opened(state, scope, active, **kwargs):
        if active.any():
            expected = incoming[-1]
            actual = STMChunks.stack_view(state)
            for column in (STMChunks.ROWS, STMChunks.ORDERS):
                torch.testing.assert_close(actual[column][active, 0], expected[column][active, 0], rtol=0, atol=0)
            value = expected[0][:, 0].clone()
            start = model.symbolSpace.ltm_store.when_start
            if start is not None:
                value[:, start-4:start+4] = actual[0][:, 0, start-4:start+4]
            torch.testing.assert_close(actual[0][active, 0], value[active], rtol=0, atol=0)
            observed.extend(active.nonzero().flatten().tolist())
        return journal(state, scope, active, **kwargs)

    monkeypatch.setattr(ShortTermMemory, 'functional_push_step_masked', staticmethod(deposit))
    monkeypatch.setattr(ChunkJournal, 'push', opened)
    try:
        with torch.no_grad():
            model.understand(model.inputSpace.prepInput(['hi no', 'we go']))
        assert observed
    finally:
        model.End()


def test_negated_sigma_reads_keep_unsigned_word_deposits_and_ideas(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from Layers import SigmaConceptsFromPercepts
    from Memory import ShortTermMemory
    from Assessment import preserved, reading_forest
    from Comprehension import Evaluation
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._tensor_peer_while_eager = True
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['hi no', 'we go'])
    pushed, ideas, trees = [], [], []
    push = ShortTermMemory.functional_push_step_masked
    finish = Evaluation.finish
    sigma = SigmaConceptsFromPercepts.from_coefficients

    def deposit(state, value, active, *args, **kwargs):
        live = value[active.reshape(-1)].detach().clone()
        if live.numel():
            assert torch.isfinite(live).all() and (live >= 0).all()
            pushed.append(live)
        return push(state, value, active, *args, **kwargs)

    def evaluated(self, *args, **kwargs):
        result = finish(self, *args, **kwargs)
        value = self.result.end_state.detach().clone()
        assert torch.isfinite(value).all() and (value >= 0).all(), {
            'negative_coordinates': (value < 0).nonzero().tolist(),
            'negative_values': value[value < 0].tolist(),
            'when_start': model.symbolSpace.ltm_store.when_start,
        }
        ideas.append(value)
        trees.append(reading_forest(model.languageSpace, self.program,
                                   range(len(self.program.leaves))))
        return result

    monkeypatch.setattr(ShortTermMemory, 'functional_push_step_masked', staticmethod(deposit))
    monkeypatch.setattr(Evaluation, 'finish', evaluated)
    observations = []
    try:
        for negate in (False, True):
            with preserved(model), monkeypatch.context() as patch:
                pushed.clear(); ideas.clear(); trees.clear()
                if negate:
                    patch.setattr(SigmaConceptsFromPercepts, 'from_coefficients',
                                  staticmethod(lambda *a, **kw: -sigma(*a, **kw)))
                model.understand(model.inputSpace.prepInput(['hi no', 'we go']))
                store = model.symbolSpace.ltm_store
                kept = store.slots[:len(store)].detach().clone()
                assert torch.isfinite(kept).all() and (kept >= 0).all()
                assert pushed and ideas
                observations.append((tuple(pushed), tuple(ideas), tuple(trees), kept))
        def same(left, right):
            if torch.is_tensor(left):
                return torch.equal(left, right)
            return len(left) == len(right) and all(same(a,b) if isinstance(a,(tuple,list)) or torch.is_tensor(a)
                else a == b for a,b in zip(left,right))
        print(dict(negated_read_equality={name: same(before,after) for name,before,after
            in zip(('deposits','ideas','trees','kept_rows'), *observations)}))
    finally:
        model.End()


def test_comprehension_stages_native_values_without_mutating_words_or_mtm(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    model._tensor_peer_while_eager = True
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
    from nomination_fixtures import nominated_sentences; nominated_sentences(model, ['hi no', 'we go'])
    ideas = []
    from Comprehension import Evaluation
    begin, finish = Evaluation.__init__, Evaluation.finish
    compose = Evaluation.composition
    def starting(self, *args, **kwargs):
        begin(self, *args, **kwargs)
        tree = self.program.tree
        self._word_tree_before = (tree.trace.leaves.clone(), tree.trace.frames.clone(),
                                  self.program.actions.clone(), tree.forest(), repr(tree.trace.marks))
        self._comprehended_points = {}
    def constituent(self, *args, **kwargs):
        value = compose(self, *args, **kwargs)
        self._comprehended_points[id(value.point)] = value.point.detach().clone()
        return value
    def arrival_points(clause):
        from ReferenceParticipants import arrival_members
        rules = model.symbolSpace.ltm_store.__dict__.get('_part_read_rules', ())
        candidates, leaves, terms = arrival_members(clause, rules)
        offered = [node.composition for node in candidates if node.composition is not None]
        offered.extend(terms[key] for key in leaves)
        return [term.point for term in offered if term.point is not None]
    def finished(self, **kwargs):
        from Comprehension import detached
        mtm = self.owner.mtm
        before_mtm = detached(mtm.state())
        result = finish(self, **kwargs)
        def unchanged(actual, before):
            if torch.is_tensor(actual):
                torch.testing.assert_close(actual, before, rtol=0, atol=0)
            elif isinstance(actual, dict):
                assert actual.keys() == before.keys()
                for key in actual:
                    unchanged(actual[key], before[key])
            elif isinstance(actual, (tuple, list)):
                assert len(actual) == len(before)
                for a, b in zip(actual, before):
                    unchanged(a, b)
            else:
                assert actual == before
        unchanged(mtm.state(), before_mtm)
        assert not hasattr(result, 'held_arrivals')
        for point in arrival_points(result.clause):
            if id(point) in self._comprehended_points:
                torch.testing.assert_close(point, self._comprehended_points[id(point)], rtol=0, atol=1e-6)
            else:
                assert any(torch.equal(point, result.store.slots[row]) for row in range(len(result.store)))
        tree = self.program.tree
        for before, after in zip(self._word_tree_before[:3],
                                (tree.trace.leaves, tree.trace.frames, self.program.actions)):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        assert (tree.forest(), repr(tree.trace.marks)) == self._word_tree_before[3:]
        return result
    monkeypatch.setattr(Evaluation, '__init__', starting)
    monkeypatch.setattr(Evaluation, 'composition', constituent)
    monkeypatch.setattr(Evaluation, 'finish', finished)
    observe = model._sentence_observation
    def completed(*args, **kwargs):
        result = observe(*args, **kwargs)
        for entry in (result.get('comprehensions', ()) if kwargs.get('record') is not None else ()):
            if entry is None:
                continue
            from Memory import OCCURRENCE, RELATION
            tree, store = entry.source.tree, entry.store
            assert store is tree.ideas
            assert not hasattr(tree, 'values')
            # Closing metadata may describe the selected operations, but
            # only native semantic entries retain their numerical values.
            from ClauseRow import Composition
            from dataclasses import fields, is_dataclass
            seen = set()
            def semantic_points(value):
                if id(value) in seen:
                    return
                seen.add(id(value))
                if isinstance(value, Composition) and value.point is not None:
                    native = any(torch.equal(value.point, store.slots[row]) for row in range(len(store)))
                    held = any(value.point is point for point in arrival_points(entry.clause))
                    assert native or held
                if is_dataclass(value):
                    for field in fields(value):
                        semantic_points(getattr(value, field.name))
                elif isinstance(value, (tuple, list)):
                    for child in value:
                        semantic_points(child)
                elif isinstance(value, dict):
                    for child in value.values():
                        semantic_points(child)
            semantic_points(entry.clause)
            pending = [entry.root]
            while pending:
                kind, reference = pending.pop()
                if kind == RELATION:
                    pending.extend(store.parts.references(reference))
                elif kind == OCCURRENCE and store.index_of_row(reference) is not None:
                    address = ('ltm', store.address_domain, reference)
                    point = model.conceptualSpace.stm.strata.read(address)
                    torch.testing.assert_close(point, store.point_of_row(address), rtol=0, atol=0)
                    ideas.append(address)
        return result
    monkeypatch.setattr(model, '_sentence_observation', completed)
    try:
        with torch.no_grad():
            model.understand(model.inputSpace.prepInput(['hi no', 'we go']))
        assert ideas
    finally:
        model.End()


def test_relative_comprehension_holds_two_ideas_and_the_native_pair():
    from test_reference_close import fixture
    from ClauseRow import Clause, Composition, ClausePredicate, predicate_point
    from Meaning import ConceptualMeaning
    from Memory import OCCURRENCE, RELATION
    from PartClosing import PartClosing
    from SentenceStrata import IdeaStack
    resolver, source, points = fixture()
    owner = resolver.components.owner
    left, right = (Composition(points[i]) for i in (0, 1))
    terms = (left, right)
    composition = Composition(right.point, terms, operation='part')
    children = tuple(Clause(ConceptualMeaning.from_description(term.point),
                           point=term.point, composition=term) for term in terms)
    predicate = ClausePredicate(predicate_point('part', points[0]), 'part')
    meaning = ConceptualMeaning(torch.stack((left.point, predicate.point, right.point)),
                                torch.ones(3, dtype=torch.bool))
    clause = Clause(meaning, relation='part', refs=(('clause', 0), predicate, ('clause', 1)),
                    children=children, composition=composition, evidence=(.7, .2))
    stack = IdeaStack(source.nDim, source.capacity, relation_capacity=source.parts.capacity)
    tree = SimpleNamespace(roots=(100,), semantic_start=101, write_edges=lambda parts: None)
    stack.prepare(source, tree, {id(left): 90, id(right): 91, id(composition): 100})
    before = {name: value.clone() for name, value in source.state_dict().items()}
    next_id = int(owner._concept_allocator.next_id)
    pending = dict(owner.mtm.pending)
    closing = PartClosing(owner, source, working=stack)
    root, _ = closing.stage(clause)
    assert root[0] == RELATION
    assert len(stack) == 2 and len(stack.parts) == 1
    assert stack.parts.references(root[1]) == ((OCCURRENCE, 90), (OCCURRENCE, 91))
    assert int(owner._concept_allocator.next_id) == next_id and owner.mtm.pending == pending
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    kept = source.write_clause(clause)
    assert len(source) == len(stack)
    torch.testing.assert_close(source.slots[:len(source)], stack.slots[:len(stack)], rtol=0, atol=0)
    torch.testing.assert_close(source.trust[:len(source)], stack.trust[:len(stack)], rtol=0, atol=0)
    torch.testing.assert_close(source.parts.truth_plus[kept[1]], stack.parts.truth_plus[root[1]], rtol=0, atol=0)
    torch.testing.assert_close(source.parts.truth_minus[kept[1]], stack.parts.truth_minus[root[1]], rtol=0, atol=0)


def test_consumed_arrival_is_held_for_the_trial_without_writing_live_mtm():
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from Memory import OCCURRENCE
    from PartClosing import PartClosing
    from SentenceStrata import IdeaStack
    from HeldConcepts import Evaluation
    from Comprehension import from_stack
    resolver, source, points = fixture()
    owner = resolver.components.owner
    identity = known_concept(owner, points[0], order=0)
    leaf = Composition(points[0]*.4, reference=identity, word_id=identity, leaves=(0,))
    raised = Composition(points[1], (leaf,), operation='lift', head_role=1, order=1)
    root = Composition(points[1], (raised,), operation='bind', head_role=1, order=1)
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    order=1, composition=root, evidence=(.7, 0.))
    stack = IdeaStack(source.nDim, source.capacity, relation_capacity=source.parts.capacity)
    tree = SimpleNamespace(roots=(100,), semantic_start=101, write_edges=lambda parts: None)
    stack.prepare(source, tree, {id(leaf): 98, id(raised): 99, id(root): 100})
    before = owner.mtm.state()
    closing = PartClosing(owner, source, working=stack)
    closing.stage(clause)
    held = Evaluation(None, SimpleNamespace())
    arrivals = held.retain_arrivals(clause, (), closing.staged_points)
    assert set(arrivals.values) == {id(leaf)}
    torch.testing.assert_close(arrivals.values[id(leaf)], leaf.point, atol=1e-6, rtol=0)
    completed = from_stack(clause, {**closing.staged_points, **arrivals.values})
    _, candidates = resolver._arrival_constituents(completed)
    assert len(candidates) == 2
    torch.testing.assert_close(candidates[1].point, leaf.point, atol=1e-6, rtol=0)
    assert completed.composition.children[0].point is None
    assert len(stack) == 1 and len(stack.parts) == 0 and len(source) == 0
    torch.testing.assert_close(owner.mtm.state(), before, rtol=0, atol=0)


def test_working_relation_reads_the_existing_pair_without_reasserting_it():
    from test_reference_close import fixture, known_concept
    from ClauseRow import Clause, Composition, ClausePredicate, predicate_point
    from Meaning import ConceptualMeaning
    from Memory import OCCURRENCE, RELATION
    from PartClosing import PartClosing
    from SentenceStrata import IdeaStack
    resolver, source, points = fixture()
    owner = resolver.components.owner
    ids = [known_concept(owner, points[i], order=0) for i in (0, 1)]
    row = source.parts.add(*ids, truth_plus=.8, truth_minus=.1)
    left, right = (Composition(points[i], reference=ids[i]) for i in (0, 1))
    predicate = ClausePredicate(predicate_point('part', points[0]), 'part')
    clause = Clause(ConceptualMeaning(torch.stack((left.point, predicate.point, right.point)),
        torch.ones(3, dtype=torch.bool)), relation='part', refs=(ids[0], predicate, ids[1]),
        composition=Composition(right.point, (left, right), operation='part'), evidence=(.3, 0.))
    stack = IdeaStack(source.nDim, source.capacity, relation_capacity=source.parts.capacity)
    tree = SimpleNamespace(roots=(100,), semantic_start=101, write_edges=lambda parts: None)
    stack.prepare(source, tree, {})
    root, _ = PartClosing(owner, source, working=stack).stage(clause)
    assert root == (RELATION, row) and len(stack) == 0
    torch.testing.assert_close(stack.parts.truth_plus[row], source.parts.truth_plus[row], rtol=0, atol=0)
    torch.testing.assert_close(stack.parts.truth_minus[row], source.parts.truth_minus[row], rtol=0, atol=0)


def test_5_comprehension_reads_the_final_preferred_object(monkeypatch):
    from test_word_interpretation import _operator
    from order_fixtures import force_order
    from Understanding import Reading, ConceptualField
    from Extension import reading
    from Comprehension import Evaluation
    owner, interpret = _operator()
    word = recurrent_word(interpret, [7], [1], form='wug')
    objects = []
    for _ in range(2):
        identity = owner.new_concept()
        force_order(owner, identity, 0)
        owner._csw_concept_row(0, identity)
        interpret.define(word, identity)
        objects.append(identity)
    owner._expected_knowing = torch.zeros_like(owner._expected_knowing)
    owner._expected_knowing[0, owner._concept_row_of(objects[0]), 0] = 1
    assert interpret.preferred(word) == (objects[0],)
    row = owner._concept_row_of(word)
    point = owner.similarity_codebook.lookup_rows(torch.tensor(row), stream=0)
    program = Reading(rows=torch.tensor([row]), word_rows=torch.tensor([row]),
        activations=torch.ones(1), leaves=point[None], actions=torch.tensor([[0, -1, 0]]),
        targets=torch.tensor([0]), end_state=torch.stack((point, point*0, point*0)),
        concept_ids=torch.tensor([word]), word_ids=torch.tensor([word]), leaf_orders=torch.tensor([0]),
        leaf_evidence=torch.tensor([[1., 0.]]), operation_values=point.new_zeros(1, 3, len(point)))
    # The sentence changes expectation after arrival, before its evaluation.
    owner._expected_knowing.zero_()
    owner._expected_knowing[0, owner._concept_row_of(objects[1]), 0] = 1
    language = SimpleNamespace(_tree_layer=lambda _: SimpleNamespace(ops=(), unary_ops=()))
    model = SimpleNamespace(_concept_owner=lambda: owner, languageSpace=language,
        symbolSpace=SimpleNamespace(ltm_store=SimpleNamespace(when_start=None)))
    field = ConceptualField(torch.tensor([word]), torch.tensor([[[[1., 0.]]]]),
        torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    evaluation = Evaluation(model, program, 0)
    reading(model, program, field, 0, torch.tensor([0]), evaluation=evaluation)
    assert evaluation.result.concept_ids.tolist() == [objects[1]]
    atoms = owner.similarity_codebook.lookup_rows(torch.tensor(owner._concept_row_of(objects[1])), stream=0)
    expected = interpret.forward(point, object_atoms=atoms, presence=torch.tensor(1.), evidence=torch.tensor([1., 0.]))
    torch.testing.assert_close(evaluation.result.end_state[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(program.leaves[0], point, rtol=0, atol=0)


def test_reference_witness_records_a_same_address_bank_read():
    from Interpret import InterpretLayer
    from ReferenceContext import ReferenceBank, prepare_operands
    value = torch.ones(1, 1, 4)
    ids = torch.tensor([[7]])
    zero = torch.zeros_like(ids)
    bank = ReferenceBank(ids, value*.5, zero.bool().logical_not(), zero.bool(),
        value[:, 0], zero[:, 0].bool(), orders=zero,
        associations=torch.tensor([[7, 7]]), interpreter=InterpretLayer())
    rule = SimpleNamespace(reference_roles=())
    result = prepare_operands(value, ids, zero, zero, orders=zero, rules=(), unary_rules=(rule,),
        bank=bank, live=(value, ids, zero, zero.bool(), zero), active=zero.bool().logical_not())
    valid = result['unary_valid'][0, 0]
    assert result['unary_refs'][0, 0, valid, 0].tolist() == [7]
    assert result['unary_resolved'][0, 0, valid, 0].tolist() == [True]
    assert result['unary_requested_orders'][0, 0, valid, 0].tolist() == [-1]
    torch.testing.assert_close(result['unary'][0, 0, valid][0], value[0, 0]*.5)


def test_comprehension_uses_the_observed_words_form_for_a_higher_order_object(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from order_fixtures import force_order
    from Understanding import Reading
    from Comprehension import Evaluation
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    try:
        owner = model._concept_owner()
        word = recurrent_word(owner.interpret, [7], [1], form='wug')
        obj = owner.new_concept()
        force_order(owner, obj, 1)
        row = owner._csw_concept_row(1, obj)
        owner.interpret.define(word, obj)
        word_row = owner._concept_row_of(word)
        bank = owner.similarity_codebook
        point = bank.lookup_rows(torch.tensor(word_row), stream=0)
        raw = bank.lookup_rows(torch.tensor(row), stream=0).clone()
        atoms = owner.interpret.lexical_atoms(torch.tensor(row), raw,
            word_ids=torch.tensor(word), stream=0)
        width = bank.mereology.percept_event_width
        assert atoms[:width].ne(raw[:width]).any()
        program = Reading(rows=torch.tensor([word_row]), word_rows=torch.tensor([word_row]),
            activations=torch.tensor([.75]), leaves=point[None], actions=torch.tensor([[0, -1, 0]]),
            targets=torch.tensor([0]), end_state=torch.stack((point, point*0, point*0)),
            concept_ids=torch.tensor([word]), word_ids=torch.tensor([word]), leaf_orders=torch.tensor([0]),
            leaf_evidence=torch.tensor([[1., 0.]]), operation_values=point.new_zeros(1, 3, len(point)))
        evaluation = Evaluation(model, program, 0)
        assert evaluation.leaf(0) == obj
        expected = owner.interpret.forward(point, object_atoms=atoms,
            presence=program.activations[0], evidence=program.leaf_evidence[0])
        torch.testing.assert_close(evaluation.leaves[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(bank.lookup_rows(torch.tensor(row), stream=0), raw, rtol=0, atol=0)
    finally:
        model.End()


def test_pending_constituent_with_unknown_order_closes_without_allocating_an_order(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from Understanding import Reading
    from ClauseJournal import finish_clause
    from Models import _boundary_registry
    model = _tiny_canonical_model(tmp_path, monkeypatch)
    try:
        owner, language = model._concept_owner(), model.languageSpace
        words = [recurrent_word(owner.interpret, [i], [1], form=form)
                 for i, form in ((7, 'wug'), (8, 'dax'))]
        assert all(owner.interpret.no_object(word) for word in words)
        rows = torch.tensor([owner._concept_row_of(word) for word in words])
        points = owner.similarity_codebook.lookup_rows(rows, stream=0)
        local = next(i for i, rule in enumerate(language._compose_binary_rules) if rule.method_name == 'part')
        frames = points.new_zeros(3, 3, points.shape[-1])
        frames[2] = torch.stack((*points, points.mean(0)))
        program = Reading(rows=rows, word_rows=rows, activations=torch.ones(2), leaves=points,
            actions=torch.tensor([[0, -1, 0], [0, -1, 1], [1, local, -1]]),
            targets=torch.tensor([0, 1]), end_state=frames[2],
            concept_ids=torch.tensor(words), word_ids=torch.tensor(words), leaf_orders=torch.full((2,), -1),
            operation_values=frames)
        next_id = int(owner._concept_allocator.next_id)
        clause = finish_clause(language, program, registry=_boundary_registry(model))
        assert clause.relation == 'part' and all(child.order == 0 for child in clause.children)
        assert program.leaf_orders.tolist() == [-1, -1]
        assert int(owner._concept_allocator.next_id) == next_id
    finally:
        model.End()


def test_comprehension_refresh_reads_its_own_published_concept():
    from Memory import STMChunks as C
    from ChunkJournal import ChunkJournal, publication_value
    from HeldConcepts import snapshot, Evaluation as HeldEvaluation
    state = C.empty(1,8,4,device='cpu',dtype=torch.float32)
    active = torch.tensor([True])
    initial = torch.tensor([[.2,.4,.1,.3]])
    state,_ = C.deposit(state,initial,active,identity=7,timeline=0)
    model = SimpleNamespace(languageSpace=SimpleNamespace(chunk_journal=SimpleNamespace(percept_band=None)))
    comprehension = HeldEvaluation(snapshot(state).for_stream(0),model)
    state,slot = C.deposit(state,initial*.5,active,identity=7,timeline=0,stack=True)
    state,actual,read = C.refresh(state,slot[:,None],active[:,None],witness=True)
    torch.testing.assert_close(comprehension.read(read[0,0]),actual[0,0])
    journal = ChunkJournal((),())
    # The read and comprehension take the same publication, with different values.
    native = torch.tensor([[.8,.6,.2,.3]])
    state,write = journal.publish(state,torch.tensor([7]),torch.tensor([0]),native,active,witness=True)
    result = torch.tensor([.3,.2,.6,.1])
    operands = torch.tensor([[.1,.4,.1,.7],[.2,.1,.1,.1]])
    record = torch.cat((write[0],torch.tensor([3,-1])))  # bind's joined description
    comprehension.publish(record,operands,result,torch.zeros(1,3,4),SimpleNamespace(where=None,when=None))
    state,actual,later = C.refresh(state,slot[:,None],active[:,None],witness=True)
    expected = publication_value(3,operands,result,result)
    torch.testing.assert_close(comprehension.read(later[0,0]),expected)
    assert not torch.equal(comprehension.read(later[0,0]),actual[0,0])
    # An S modifier's restoration reads the earlier converted operand.
    earlier = torch.tensor([[[.9,.2,.3,.4],[0.,0.,0.,0.],[0.,0.,0.,0.]]])
    comprehension.publish(torch.cat((write[0],torch.tensor([1,0]))),operands,result,earlier,
                       SimpleNamespace(where=None,when=None))
    torch.testing.assert_close(comprehension.read(later[0,0]),earlier[0,0])
