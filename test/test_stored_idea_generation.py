"""Stored ideas invert from dictionaries and live wholes, without input traces."""
from types import MethodType, SimpleNamespace

import torch
import pytest

from DecompositionChooser import DecompositionChooser
from Language import LanguageSpace


def additive_language():
    class Add:
        inverse_kind = 'residual'
        residual_scale = 1.
        def compose(self, left, right):
            return left + right
    language = SimpleNamespace(_generate_binary_ops=(Add(),), _generate_unary_ops=(),
        decomposition_chooser=DecompositionChooser())
    for name in ('reverse_binary_step', '_reverse_of_binary_op', '_finish_binary_inverse',
                 'generate_unary_step', 'choose_generate'):
        setattr(language, name, MethodType(getattr(LanguageSpace, name), language))
    language._bounded_binary_reconstruction = LanguageSpace._bounded_binary_reconstruction
    language.generate_policy_logits = lambda value, features=None: value.new_tensor([1., 0.]).expand(len(value), -1)
    return language


def test_stored_unfold_uses_forward_search_and_terminal_eligibility():
    from MemoryIndex import unfold_idea
    basis = torch.eye(4)
    result = unfold_idea(additive_language(), basis, basis[0] + basis[1], 8,
                         activation=torch.full((4,), 2.))
    assert result['complete']
    assert result['codes'] == (0, 1)
    assert result['spent'] == 3


def test_missing_compound_is_incomplete_instead_of_losing_a_phrase():
    from MemoryIndex import unfold_idea
    words = torch.eye(4)
    result = unfold_idea(additive_language(), words, words[:3].sum(0), 8,
                         activation=torch.full((4,), 2.))
    assert not result['complete']
    assert result['codes'] == ()


@pytest.mark.parametrize('compiled', [False, True])
def test_stationary_pending_walk_keeps_the_full_budget_result_and_pullback(compiled, monkeypatch):
    import Models
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    owner = SimpleNamespace(languageSpace=additive_language())
    words = torch.eye(4, dtype=torch.float64)[None, :2].expand(2, -1, -1).clone()
    event = torch.zeros(2, 4, 4, dtype=torch.float64)
    event[0, 0] = words[0].sum(0)
    event[1, 0, 3] = 1.  # This top has no legal inverse or terminal.
    event.requires_grad_()
    def walk(value):
        return Models.BasicModel._output_generate_walk(owner, value, 12,
            basis=words, basis_valid=torch.ones(2, 2, dtype=torch.bool),
            sample_actions=True, return_candidates=True,
            terminal_valid=torch.tensor([[True,True],[False,False]]))
    ordinary_loop = Models._reconstruction_while_loop
    trips = []
    def full_budget(condition, body, values):
        while bool((values[0] < 12) & (values[2] > 0).any()):
            trips.append(1)
            values = body(*values)
        return values
    initial_rng = torch.get_rng_state()
    monkeypatch.setattr(Models, '_reconstruction_while_loop', full_budget)
    expected = walk(event)
    expected_rng = torch.get_rng_state()
    expected_gradient, = torch.autograd.grad(expected[0].sum(), event)
    assert len(trips) == 12
    monkeypatch.setattr(Models, '_reconstruction_while_loop', ordinary_loop)
    torch.set_rng_state(initial_rng)
    decode = torch.compile(walk, fullgraph=True, backend='eager') if compiled else walk
    actual = decode(event)
    actual_gradient, = torch.autograd.grad(actual[0].sum(), event)
    for a, b in zip(actual[:-1], expected[:-1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # The diagnostic full-budget loop also visits the stationary suffix;
    # only actual decisions supply a policy-credit input.
    decisions = actual[4] >= 0
    torch.testing.assert_close(actual[-1][decisions], expected[-1][decisions], rtol=0, atol=0)
    assert not actual[-1].requires_grad
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert actual[1].tolist() == [2, 0] and actual[2].tolist() == [False, True]


def test_stored_unfold_preserves_live_native_operator_activation():
    from Language import LiftLayer
    from MemoryIndex import unfold_idea
    language = additive_language()
    op = LiftLayer(nInput=4, nOutput=4)
    language._generate_binary_ops = (op,)
    words = torch.eye(4) * .1
    root = op.compose(words[0:1], words[1:2])[0].detach()
    prior = torch.full((1, 4), 7.)
    op._sigma.activation = prior
    result = unfold_idea(language, words, root, 8, activation=torch.full((4,), 2.))
    assert result['complete'] and result['codes'] == (0, 1)
    assert op._sigma.activation is prior
    torch.testing.assert_close(op._sigma.activation, torch.full((1, 4), 7.))


@pytest.mark.parametrize('compiled', [False, True])
def test_compound_candidates_are_not_word_emissions(compiled):
    from Models import BasicModel
    language = additive_language()
    owner = SimpleNamespace(languageSpace=language)
    words = torch.eye(4)[None]
    phrase = (words[:, 0] + words[:, 1])[:, None]
    event = torch.zeros(1, 5, 4)
    event[:, 0] = phrase[:, 0] + words[:, 2]
    def walk(event, words, phrase):
        return BasicModel._output_generate_walk(owner, event, 8,
            basis=words, basis_valid=torch.ones(1, 4, dtype=torch.bool),
            constituents=phrase, constituent_valid=torch.ones(1, 1, dtype=torch.bool))
    decode = torch.compile(walk, fullgraph=True, backend='inductor') if compiled else walk
    out, count, truncated, _ = decode(event, words, phrase)
    assert count.tolist() == [3] and not truncated.any()
    # Commutative operations cannot recover a unique surface order. Every
    # emitted object must still be a leaf, and all three leaves must survive.
    torch.testing.assert_close(out[:, :3].sum(1), event[:, 0])
    assert all(any(torch.equal(leaf, word) for word in words[0]) for leaf in out[0, :3])


@pytest.mark.parametrize('reference', [('meta', 1), ('sym', -1), ('sym', 0),
                                      ('sym', -42), ('sym', True), ('sym', 1.5)])
def test_nontaxonomic_cue_does_not_enter_concept_order_reader(reference):
    from Layers import LTM0
    from Meaning import ConceptualMeaning
    from Taxonomy import concept_reference
    store = LTM0(4, capacity=2)
    cue = ConceptualMeaning(torch.ones(3, 4), torch.tensor([True, False, False]),
                            role_refs=(reference, None, None))
    # Content reads do not unfold or send reference-shaped values to an
    # order reader. An empty store supplies no result for any cue domain.
    result = store.cued_rows(cue)
    assert result['value'] == () and result['records_scanned'] == 0


@pytest.mark.parametrize('compiled', [False, True])
def test_every_family_can_occupy_either_side_when_recomposition_fits(compiled):
    from Generative import inverse_menu
    language = additive_language()
    class Ordered:
        inverse_kind = 'search'
        def compose(self, left, right):
            return left + 2 * right
    language._generate_binary_ops = (Ordered(),)
    language._generate_binary_rules = (SimpleNamespace(clause_form='S'),)
    noun = torch.tensor([1., 0., 1., 0.])
    verb = torch.tensor([0., 1., 0., 1.])
    types = torch.stack((verb, noun))[None].expand(16, -1, -1).clone()
    # Exercise every family pair and both numerical orientations. Copies in
    # priming and STM have the same eligibility as the original candidates.
    basis = torch.cat((torch.eye(4)[:2][None].expand(16, -1, -1), types), 1)
    constituents = torch.cat((types, types), 1)
    families = torch.tensor([[3, 3, a, b] for a in range(4) for b in range(4)])
    reverse = torch.arange(16).remainder(2).bool()[:, None]
    expected_left = torch.where(reverse, noun, verb)
    expected_right = torch.where(reverse, verb, noun)
    def menu(parent, basis, constituents, families):
        return inverse_menu(language, parent, torch.ones(16, dtype=torch.bool),
            basis=basis, basis_valid=torch.ones(16, 4, dtype=torch.bool),
            terminal_valid=torch.tensor([[True, True, False, False]]).expand(16, -1),
            constituents=constituents, constituent_valid=torch.ones(16, 4, dtype=torch.bool),
            constituent_families=families)
    decode = torch.compile(menu, fullgraph=True, backend='inductor') if compiled else menu
    left, right, legal = decode(expected_left + 2 * expected_right, basis, constituents, families)[:3]
    assert legal.tolist() == [[True, False]] * 16
    torch.testing.assert_close(left[:, 0], expected_left)
    torch.testing.assert_close(right[:, 0], expected_right)


def test_live_inverse_snapshot_is_bounded_detached_and_excludes_singletons():
    from Generative import LiveConstituents
    words = torch.eye(4)[None]
    snapshot = LiveConstituents(words, torch.ones(1, 4, dtype=torch.bool), 2)
    for n in (1, 2, 3, 4):
        value = words[:, :n].sum(1).requires_grad_()[:, None]
        import STMChunks as chunks
        state,_ = chunks.deposit(chunks.empty(1,8,4),value[:,0],torch.tensor([True]),stack=True)
        support = torch.zeros(1,8,4,dtype=torch.bool)
        support[:,0] = torch.tensor([i < n for i in range(4)])
        snapshot.observe(state, torch.tensor([True]), support)
        assert not snapshot.codes.requires_grad
    assert snapshot.valid.tolist() == [[True, True]]
    torch.testing.assert_close(snapshot.codes[0, 0], words[0, :3].sum(0))
    torch.testing.assert_close(snapshot.codes[0, 1], words[0].sum(0))
    assert set(snapshot.__dict__) == {'words', 'word_valid', 'codes', 'valid'}


def test_sentence_fork_carries_each_rows_inverse_prefix_without_aliasing(monkeypatch):
    from Generative import LiveConstituents
    from SentenceFork import SentenceFork
    words = torch.zeros(2, 1, 4)
    field = LiveConstituents(words, torch.zeros(2, 1, dtype=torch.bool), 2)
    field.codes[:, -1] = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    field.valid[:, -1] = True
    fork = SentenceFork(torch.tensor([True, True]))
    fork.inverse_candidates = field
    draws = iter((torch.tensor([0., 0.]), torch.tensor([.9, .1])))
    monkeypatch.setattr(torch, 'rand_like', lambda *args, **kwargs: next(draws))
    choice = SimpleNamespace(valid=torch.tensor([True, True]),
        alternative_count=torch.ones(2, dtype=torch.long), action=torch.zeros(2, dtype=torch.long))
    fork.record(choice, torch.tensor(0), (words,), word=torch.tensor(0), latches=())
    field.codes[:, -1] += 10
    fork.record(choice, torch.tensor(1), (words,), word=torch.tensor(1), latches=())
    # Row 0 departs at the first choice, row 1 at the second. Neither may
    # inherit a constituent formed later in the greedy continuation.
    field.codes[:, -1] += 10
    assert fork.slot.tolist() == [0, 1]
    assert fork.inverse_prefix[0][:, -1].tolist() == [[1., 2., 3., 4.], [15., 16., 17., 18.]]
    assert not fork.inverse_prefix[0].requires_grad
    trial = LiveConstituents(words, field.word_valid, 2)
    trial.restore(field.snapshot(), torch.tensor([False, True]))  # bind-only row
    trial.restore(fork.inverse_prefix, torch.tensor([True, False]))  # compose departure
    assert trial.codes[:, -1].tolist() == [[1., 2., 3., 4.], [25., 26., 27., 28.]]
    assert trial.valid.tolist() == [[False, True], [False, True]]


def test_type_family_snapshot_uses_learned_membership_and_row_local_priming():
    from Generative import primed_type_families
    codes = torch.eye(5)
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(W=codes, lookup_rows=lambda rows,**kw: codes[rows]),
        components=SimpleNamespace(individual_ids=(101,), verbs=SimpleNamespace(ids=(102,))),
        concept_id_at_row=lambda row: 100 + row, _row_order=lambda row: int(row > 0))
    heat = torch.tensor([[2., 3., 1., 4., 1.], [3., 1., 4., 1., 5.]])
    result = primed_type_families(owner, heat, limit=1)
    assert result.valid.tolist() == [[True, False, True], [False, True, True]]
    torch.testing.assert_close(result.codes[0, 0], codes[1])
    torch.testing.assert_close(result.codes[1, 1], codes[2])
    torch.testing.assert_close(result.codes[0, 2], codes[3])
    torch.testing.assert_close(result.codes[1, 2], codes[4])
    assert result.families.tolist() == [0, 1, 2]


def determiner_language(words):
    from Language import LowerLayer
    rules = [SimpleNamespace(determiner_mode=mode, key=i) for i, mode in enumerate(('mint', 'bind'))]
    seen = []
    def score(pairs, folded, _stop, _reduce, *, op_indices):
        seen.append((pairs.clone(), folded.clone()))
        # Stand-in learned weights. Renaming/reordering candidates cannot
        # turn a word identity into a grammar lookup.
        weights = words[int(op_indices[0])]
        return None, (pairs[:, 0] @ weights)[:, None, None]
    layer = SimpleNamespace(d_model=4, chooser=SimpleNamespace(score_binary=score),
        stop_anchor=torch.zeros(1, 4), reduce_anchor=torch.zeros(2, 4))
    language = additive_language()
    language._generate_binary_rules = language._compose_binary_rules = rules
    language._generate_rule_key = lambda rule, arity: rule.key
    language.language_layer = SimpleNamespace(operation_layer=layer)
    language._generate_binary_ops = (LowerLayer(4, 4), LowerLayer(4, 4))
    language.generate_policy_logits = lambda value, features=None: value.new_tensor([2., 1., 0.]).expand(len(value), -1)
    return language, seen


def test_determiner_uses_order_and_identity_with_learned_marker_scores():
    from Generative import determiner_expansions
    words = torch.eye(4)
    language, seen = determiner_language(words)
    orders = torch.tensor([0, 0, 2, 0])
    for mode, marker in (('mint', 0), ('bind', 1)):
        result = determiner_expansions(language, words[2], 1, words, orders, binding=mode)
        assert tuple(result) == (marker,)
        left, right, lo, ro = result[marker]
        torch.testing.assert_close(left, words[marker])
        torch.testing.assert_close(right, words[2])
        assert (lo, ro) == (0, 2)
    assert not determiner_expansions(language, words[2], 2, words, orders)
    assert len(seen) == 2


def test_stored_order_mismatch_expands_determiner_and_terminates():
    from MemoryIndex import unfold_idea
    words = torch.eye(4)
    language, _ = determiner_language(words)
    orders = [0, 0, 1, 0]
    result = unfold_idea(language, words, words[2], 8, order=0, binding='bind',
        activation=torch.full((4,), 2.), order_of=orders.__getitem__,
        sigma_inverse=lambda code, order: (words[3],) if code == 2 else ())
    assert result['complete']
    assert result['codes'] == (1, 2, 3)
    assert result['operations'][0] == ('binary', 1)
    assert result['spent'] == 4


def test_stored_binding_reads_filled_identity_not_formation_provenance():
    from dataclasses import replace
    from MemoryIndex import stored_binding
    from Meaning import ConceptualMeaning
    value = replace(ConceptualMeaning.from_description(torch.ones(4)),
        role_refs=(('sym', 71), None, None), bindings={'_bound_roles': (('referent', 0),)})
    assert stored_binding(value, 0) == 'bind'
    assert stored_binding(replace(value, role_refs=(None, None, None)), 0) is None
    for choice in ('mint', 'bind', 'open'):
        provenance = replace(value, bindings={'_formation_records':
            ({'role': 0, 'choice': choice, 'probability': .9, 'reference': 71},)})
        assert stored_binding(provenance, 0) is None


def test_native_catalog_can_realize_both_determiner_identity_choices(tmp_path):
    from Generative import determiner_expansions
    from math_chain_ordinary import ordinary_model
    model = ordinary_model(tmp_path)
    try:
        language = model.languageSpace
        width = language.language_layer.operation_layer.d_model
        words = torch.eye(width)[:3]
        orders = torch.tensor([0, 0, 1])
        keys = language._generate_rule_keys.tolist()
        assert len(keys) == len(set(keys))
        for mode in ('mint', 'bind'):
            choices = determiner_expansions(language, words[2], 0, words, orders, binding=mode)
            assert choices, mode
            assert all(language._generate_binary_rules[i].determiner_mode == mode for i in choices)
        # A pre-item-6 checkpoint has the mint row but no definite row.
        # Its learned weights retain their meanings when that row is added.
        definite = next(i for i, rule in enumerate(language._generate_binary_rules)
                        if rule.determiner_mode == 'bind')
        retained = [i for i in range(len(keys)) if i != definite]
        state = {name: value.detach().clone() for name, value in language.state_dict().items()}
        for name in ('_generate_rule_keys', 'generate_policy.weight', 'generate_policy.bias'):
            state[name] = state[name][retained].clone()
        saved = state['generate_policy.weight'].clone()
        migration = language.migrate_generate_checkpoint(state, '')
        assert migration['generate_policy.weight'][definite] == -1
        torch.testing.assert_close(state['generate_policy.weight'][retained], saved, rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_unnamed_primed_concept_cannot_complete_as_a_word(monkeypatch):
    # Make the default eager-capture backend explicit: a prior module must
    # not accidentally turn this HOP alias regression into an eager test.
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    from Models import BasicModel
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    codes = torch.eye(4)[:2][None]
    bank = PrimedSymbols(torch.tensor([[0, 1]]), codes, torch.ones(1, 2),
        torch.tensor([[True, False]]), torch.tensor([[[97], [0]]]),
        torch.tensor([[[True], [False]]]))
    root = codes[:, 1]
    end = torch.cat((root[:, None], root.new_zeros(1, 2, 4)), 1)
    record = SentenceUnderstanding(root, end, torch.ones(1, dtype=torch.long),
        end.flatten(1)[:, None], torch.ones(1, 1, dtype=torch.long), codes[:, :1],
        torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool),
        bank, torch.tensor(0), codes[:, 1:], torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[2]]))
    owner = SimpleNamespace(languageSpace=additive_language(),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1, 1, dtype=torch.bool)),
        _byte_tables=lambda B, W: (True, bank.bytes[:, :1], bank.byte_valid[:, :1]),
        _byte_word_cost=lambda *args, **kwargs: root.new_zeros(1))
    def decode(value, slots, depth, candidates, valid, weights, **options):
        return BasicModel._output_generate_walk(owner, slots, 8, basis=candidates,
            basis_valid=valid, basis_priming=weights, return_trace=True, **options)
    owner._decode_conceptual_sentence = decode
    result, trace = BasicModel._reconstruct_sentences(owner, root, record.word_values,
        record.roots, record.depths, end, record.end_depth, record.sentence,
        understanding=record, return_decoded=True)
    assert result[3].tolist() == [True]
    # Approximate native words may emit, but the unspelled whole cannot
    # masquerade as a lexical leaf or obtain an exact completion certificate.
    assert not trace[-5].any()
    assert not any(torch.equal(value, root[0]) for value in trace[0][0,:int(trace[1][0])])
    assert trace[3].item() > 0
    # It stays available as structure; lexical exclusion does not erase it.
    assert record.constituent_valid.all() and bank.valid.all()


def test_unspelled_primed_compound_stays_available_for_decomposition():
    from Models import BasicModel
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    words = torch.eye(4)[:3][None]
    codes = torch.cat((words, words[:, :2].sum(1, keepdim=True)), 1)
    bank = PrimedSymbols(torch.tensor([[0, 1, 2, 3]]), codes, torch.ones(1, 4),
        torch.tensor([[True, True, True, False]]), torch.tensor([[[97], [98], [99], [0]]]),
        torch.tensor([[[True], [True], [True], [False]]]))
    root = words.sum(1)
    end = torch.cat((root[:, None], root.new_zeros(1, 2, 4)), 1)
    record = SentenceUnderstanding(root, end, torch.ones(1, dtype=torch.long),
        end.flatten(1)[:, None], torch.ones(1, 1, dtype=torch.long), words,
        torch.arange(3)[None], torch.ones(1, 3, dtype=torch.bool), bank, torch.tensor(0))
    owner = SimpleNamespace(languageSpace=additive_language(),
        inputSpace=SimpleNamespace(_word_active_mask=torch.ones(1, 3, dtype=torch.bool)),
        _byte_tables=lambda B, W: (True, bank.bytes[:, :3], bank.byte_valid[:, :3]),
        _byte_word_cost=lambda *args, **kwargs: root.new_zeros(1))
    def decode(value, slots, depth, candidates, valid, weights, **options):
        return BasicModel._output_generate_walk(owner, slots, 8, basis=candidates,
            basis_valid=valid, basis_priming=weights, return_trace=True, **options)
    owner._decode_conceptual_sentence = decode
    result, trace = BasicModel._reconstruct_sentences(owner, root, words,
        record.roots, record.depths, end, record.end_depth, record.sentence,
        understanding=record, return_decoded=True)
    assert result[3].tolist() == [False] and trace[1].tolist() == [3]
    torch.testing.assert_close(trace[0][:, :3].sum(1), root)
    assert all(any(torch.equal(value, word) for word in words[0]) for value in trace[0][0, :3])


@pytest.mark.parametrize('near', [False, True])
def test_dropped_constituent_is_charged_and_never_complete(near, eager_reading, monkeypatch):
    from Models import BasicModel
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    language = additive_language()
    class Projection:
        inverse_kind = 'search'
        def compose(self, left, right):
            return right + (1e-6 * left if near else 0.)
    language._generate_binary_ops = (Projection(),)
    words = torch.eye(4)[:2][None]
    root = language._generate_binary_ops[0].compose(words[:, 0], words[:, 1])
    end = torch.cat((root[:, None], root.new_zeros(1, 2, 4)), 1)
    bank = PrimedSymbols(torch.tensor([[0, 1]]), words, torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[[97], [98]]]),
        torch.ones(1, 2, 1, dtype=torch.bool))
    record = SentenceUnderstanding(root, end, torch.ones(1, dtype=torch.long),
        end.flatten(1)[:, None], torch.ones(1, 1, dtype=torch.long), words,
        bank.rows, bank.valid, bank, torch.tensor(0))
    owner = SimpleNamespace(languageSpace=language,
        inputSpace=SimpleNamespace(_word_active_mask=bank.valid),
        _byte_tables=lambda B, W: (True, bank.bytes, bank.byte_valid),
        # Perfect head realization must still pay for the lost modifier.
        _byte_word_cost=lambda *a, **kw: root.new_zeros(1))
    owner._decode_conceptual_sentence = lambda value, slots, depth, candidates, valid, weights, **kw: (
        BasicModel._output_generate_walk(owner, slots, 8, basis=candidates,
            basis_valid=valid, basis_priming=weights, return_trace=True, **kw))
    result, trace = BasicModel._reconstruct_sentences(owner, root, words, record.roots,
        record.depths, end, record.end_depth, record.sentence, understanding=record, return_decoded=True)
    assert trace[1].tolist() == [1]
    assert trace[2].tolist() == [False]  # The stack drained.
    assert result[3].tolist() == [True]  # Dropped coverage is still a failure.
    assert result[2].tolist() == [.5]

    # Under the field objective, free_bytes is reporting-only. Verify that
    # the lost support still changes the actual trial/keep cost when the
    # derivation inverse has no defined coordinates to charge.
    import DerivationReconstruction
    monkeypatch.setattr(DerivationReconstruction, 'reconstruct', lambda *a:
        (words * 0, torch.ones(1, dtype=torch.bool), torch.zeros_like(words, dtype=torch.bool)))
    field = SimpleNamespace(scope=bank.valid, observed=words,
        costs=lambda rec, values, **kw: ({key: root.new_zeros(1)
            for key in ('inside', 'outside', 'work')}, values))
    owner.__dict__.update(reconstruct_in_loop=True, loss=SimpleNamespace(reconstruction_scale=1.),
        inter_loss_weight=0., inter_contrastive_weight=0., symbolSpace=SimpleNamespace(expectation=None),
        _publish_sentence_scratch=lambda state: None, _trial_understanding=lambda *a: record,
        _tensor_pushed_ideas=words, _sentence_field=field, _last_decoder_trace=trace,
        _sentence_leaf_positions=lambda *a: torch.arange(2), WHAT_STEP_COST=1.,
        _reconstruct_trial=lambda rec: result, _last_decoder_byte_identity=result[2],
        _sentence_observation=lambda *a: dict(entries=(None,), meanings=(None,)))
    owner.inputSpace._reconstruction_sentence_available = torch.ones(1, 1, dtype=torch.bool)
    lang = [None] * 12
    lang[6], lang[10], lang[11] = root[:, None], end.flatten(1)[:, None], record.depths
    cost, *_ = BasicModel._sentence_path_cost(owner, ((end,), lang, None), 0, torch.tensor([True]))
    assert cost.tolist() == [.5]
    terms = owner._sentence_cost_registry.breakdown()
    assert terms['reconstruction.coverage']['trained']
    assert not terms['reconstruction.free_bytes']['trained']


@pytest.mark.parametrize('available', [True, False])
def test_keep_cost_charges_erased_meaning_even_with_every_surface_leaf(available):
    from types import SimpleNamespace
    from Models import BasicModel
    from SentenceUnderstanding import PrimedSymbols, SentenceUnderstanding
    # Opposing leaves retain content even when their signed sum is zero.
    words = torch.tensor([[[1., 0., .2, 0.], [0., 1., -.2, 0.]]])
    root = words.sum(1)
    end = torch.cat((root[:, None], root.new_zeros(1, 2, 4)), 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    bank = PrimedSymbols(torch.tensor([[0, 1]]), words, torch.ones(1, 2), valid,
        torch.tensor([[[97], [98]]]), torch.ones(1, 2, 1, dtype=torch.bool), meaning_start=2)
    record = SentenceUnderstanding(root, end, torch.ones(1, dtype=torch.long),
        end.flatten(1)[:, None], torch.ones(1, 1, dtype=torch.long), words,
        bank.rows, valid, bank, torch.tensor(0))
    # Surface-perfect free leaves with no meaning: byte/count checks cannot
    # distinguish this value from a complete reading. No operator is masked.
    returned = words.clone()
    returned[..., 2:] = 0.
    owner = SimpleNamespace(reconstruct_in_loop=True, loss=SimpleNamespace(reconstruction_scale=1.),
        inter_loss_weight=0., inter_contrastive_weight=0., symbolSpace=SimpleNamespace(expectation=None),
        inputSpace=SimpleNamespace(_reconstruction_sentence_available=torch.tensor([[available]])),
        _publish_sentence_scratch=lambda state: None, _trial_understanding=lambda *a: record,
        _tensor_pushed_ideas=words, _sentence_field=None,
        _reconstruct_trial=lambda rec: (returned, root.new_zeros(1), root.new_zeros(1)),
        _last_decoder_byte_identity=root.new_zeros(1),
        _sentence_observation=lambda *a: dict(entries=(None,), meanings=(None,)))
    lang = [None] * 12
    lang[6], lang[10], lang[11] = root[:, None], record.roots, record.depths
    state = ((end,), lang, None)
    erased, *_ = BasicModel._sentence_path_cost(owner, state, 0, torch.tensor([True]))
    returned = words.flip(1)
    retained, *_ = BasicModel._sentence_path_cost(owner, state, 0, torch.tensor([True]))
    assert erased.item() == float(available)
    assert retained.item() == 0.


def test_unspelled_idempotent_whole_cannot_hide_its_lexical_inverse(monkeypatch):
    from Models import BasicModel
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'eager')
    language = additive_language()
    class Mean:
        inverse_kind = 'residual'
        residual_scale = 2.
        def compose(self, left, right):
            return (left + right) / 2
    language._generate_binary_ops = (Mean(),)
    words = torch.eye(4)[:2][None]
    root = words.mean(1, keepdim=True)
    # The unspelled whole is first and has an exact self/self inverse. Search
    # must eliminate that pair before it outranks the supported word pair.
    bank = torch.cat((root, words), 1)
    event = torch.cat((root, root.new_zeros(1, 3, 4)), 1)
    valid = torch.ones(1, 3, dtype=torch.bool)
    owner = SimpleNamespace(languageSpace=language)
    out, count, truncated, _ = BasicModel._output_generate_walk(owner, event, 8,
        basis=bank, basis_valid=valid, basis_priming=valid.float(),
        terminal_valid=torch.tensor([[False, True, True]]),
        constituents=bank[:, :1], constituent_valid=valid[:, :1])
    assert count.tolist() == [2] and not truncated.any()
    torch.testing.assert_close(out[:, :2].sum(1), words.sum(1))
