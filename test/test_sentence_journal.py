"""Numerical compose records live for one open sentence, at global trace addresses."""
from types import SimpleNamespace

import torch

from Models import BasicModel


def test_no_numerical_frames_without_a_sentence_closing_reader(tmp_path):
    from test_reverse_traversal import _traversal_model, _run
    model = _traversal_model(tmp_path)
    try:
        # Disable the consumer during the run. _run's normal boundary
        # enables it inside the context and clears it on exit.
        with model._sentence_run():
            model._sentence_ends = False
            model._sentence_reconstruction = False
            _run(model, ['1 plus 2', '3 plus 4'])
        trace = model._reconstruction_stack()
        assert not getattr(model, '_sentence_ends', False)
        assert trace._choice_mask.any(), 'the probe must perform an operation'
        assert not trace._choice_values.any(), 'no closing consumes these frames'
        assert (trace._choice_refs == -1).all()
        assert not trace._choice_ref_relations.any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_packed_journal_reuses_one_sentence_without_colliding_with_its_closing():
    active = torch.tensor([[True] * 6 + [False] * 250,
                           [True] * 5 + [False] * 251])
    ids = torch.tensor([[0, 0, 1, 1, 1, 1] + [-1] * 250,
                        [0, 0, 0, 1, 1] + [-1] * 251])
    columns, width = BasicModel._sentence_journal_layout(active, ids, 16, 4864)
    assert width == 3 * 8 + 16
    assert columns[0, :18].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert columns[1, :15].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5]
    # Intermediate and final global endings share the closing's local window.
    assert columns[:, 768:784].tolist() == [list(range(24, 40))] * 2
    assert columns[:, 800:816].tolist() == [list(range(24, 40))] * 2


def test_journal_shape_is_fixed_within_the_serial_word_bucket():
    layouts = []
    for length in (2, 3, 4, 7):
        active = torch.arange(8)[None] < length
        ids = torch.where(active, 0, -1)
        layouts.append(BasicModel._sentence_journal_layout(active, ids, 6, 72))
    assert {width for _, width in layouts} == {30}
    for columns, _ in layouts:
        assert columns[0, 24:30].tolist() == list(range(24, 30))


def test_large_packed_capacity_uses_existing_sentence_buckets():
    for length, bound in ((1, 8), (7, 8), (8, 8), (9, 16), (17, 32), (256, 256)):
        active = torch.arange(256)[None] < length
        ids = torch.where(active, 0, -1)
        columns, width = BasicModel._sentence_journal_layout(active, ids, 16, 4864)
        assert width == 3 * bound + 16
        assert columns[0, 3 * (length - 1)].item() == 3 * (length - 1)
        assert columns[0, 768].item() == 3 * bound


def test_program_reads_compact_journal_by_global_address_and_keeps_value_gradients():
    active = torch.ones(1, 4, dtype=torch.bool)
    ids = torch.tensor([[0, 0, 1, 1]])
    columns, width = BasicModel._sentence_journal_layout(active, ids, 4, 28)
    frames = torch.arange(width * 12, dtype=torch.float32).reshape(1, width, 12).requires_grad_()
    refs = torch.arange(width * 2).reshape(1, width, 2)
    relations = refs.remainder(2).bool()
    trace = SimpleNamespace(_choice_values=frames, _choice_refs=refs,
        _choice_ref_relations=relations, _choice_journal_columns=columns)
    model = SimpleNamespace(_reconstruction_stack=lambda: trace,
        inputSpace=SimpleNamespace(), symbolSpace=SimpleNamespace())
    program = (torch.tensor([[2, 3]]),
        torch.tensor([[[0, -1, 0], [0, -1, 1], [1, 0, -1]]]),
        torch.tensor([[0, 1, 1]]), torch.tensor([[-1, -1, 12]]))
    entry, = BasicModel._program_entries(model, program, torch.zeros(1, 4, 4),
        torch.arange(4)[None], torch.arange(4)[None], torch.ones(1, 4), torch.zeros(1, 3, 4))
    torch.testing.assert_close(entry.operation_values[-1], frames[0, 12].reshape(3, 4), rtol=0, atol=0)
    assert entry.operation_refs[-1].tolist() == refs[0, 12].tolist()
    assert entry.operation_relations[-1].tolist() == relations[0, 12].tolist()
    # Numerical readings of a concluded field must retain this route. There
    # is no new detach at either the compact bank or program boundary.
    entry.operation_values[-1].sum().backward()
    expected = torch.zeros_like(frames)
    expected[0, 12] = 1
    torch.testing.assert_close(frames.grad, expected, rtol=0, atol=0)


def test_inactive_columns_neither_split_a_sentence_nor_alias_its_active_words():
    active = torch.tensor([[True, False, True, True]])
    ids = torch.tensor([[0, -1, 0, 1]])
    columns, width = BasicModel._sentence_journal_layout(active, ids, 4, 28)
    assert width == 3 * 4 + 4
    assert columns[0, :3].tolist() == [0, 1, 2]
    assert columns[0, 6:9].tolist() == [3, 4, 5]
    assert columns[0, 9:12].tolist() == [0, 1, 2]


def test_packed_word_and_closing_leave_other_sentence_rows_unchanged(tmp_path, monkeypatch):
    from test_meronomy_ladder import _build_ladder_variant
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'packed_row_isolation', [
        ('<architecture>', '<architecture><composeTemperature>0.2</composeTemperature>'),
        ('<serialWordCapacity>8</serialWordCapacity>', '<serialWordCapacity>16</serialWordCapacity>'),
        ('<serialWordBuckets>8</serialWordBuckets>', '<serialWordBuckets>16</serialWordBuckets>')])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    chooser = model.languageSpace._tree_layer(2).chooser
    score_unary = chooser.score_unary
    def prefer_unary(*args, **kwargs):
        stop, unary = score_unary(*args, **kwargs)
        return stop, torch.full_like(unary, 1e6)
    monkeypatch.setattr(chooser, 'score_unary', prefer_unary)
    allowed = None
    operations, stray, changed, inactive_words = [], [], [], []
    choose = model.languageSpace.choose_operation
    def observe_choice(state, gate, **kwargs):
        result = choose(state, gate, **kwargs)
        choice = result[0] if isinstance(result, tuple) else result
        operations.append(int(choice.applied.sum()))
        if allowed is not None:
            stray.append(int((choice.applied & ~allowed).sum()))
        return result
    monkeypatch.setattr(model.languageSpace, 'choose_operation', observe_choice)
    run = model._run_sentence_word_bricks
    def observe_words(*args, **kwargs):
        compose = args[-1]
        def observe(payload, index, live, lang, stm):
            nonlocal allowed
            allowed = payload[7].reshape(-1).bool()
            inactive = ~allowed
            inactive_words.append(int(inactive.sum()))
            before = tuple(v.clone() for v in (*lang, *stm))
            result = compose(payload, index, live, lang, stm)
            for field, (old, new) in enumerate(zip(before, (*result[0], *result[1]))):
                if not torch.equal(old[inactive], new[inactive]):
                    changed.append((model._open_sentence_slot, int(index), field))
            return result
        return run(*args[:-1], observe, **kwargs)
    monkeypatch.setattr(model, '_run_sentence_word_bricks', observe_words)
    try:
        raw = model.inputSpace.prepPackedInput([['1', '2 plus 3'], ['4 plus 5']])
        with torch.no_grad():
            model(raw)
        assert sum(inactive_words) > 0
        assert sum(operations) > 0
        assert not any(stray), f'operations outside the open sentence: {stray}; changed={changed}'
        assert changed == [], f'inactive row state changed: {changed}'
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def _deep_derivation_owner(words=600):
    slots = 3*words+4
    ids = torch.full((1, slots), -1, dtype=torch.long)
    ids[:, 3:3*words:3] = 0
    ids[:, 2:3*words:3] = 1
    trace = SimpleNamespace(choices=lambda: (ids, None, ids >= 0),
                           _choice_positions=torch.zeros_like(ids))
    language = SimpleNamespace(_cs_binary_rule_ids=torch.tensor([0]),
        _cs_unary_rule_ids=torch.tensor([1]),
        language_layer=SimpleNamespace(operation_layer=SimpleNamespace(pair_rewrites=[False])),
        local_op_from_rule_ids=lambda values, catalog:
            (torch.zeros_like(values), values == catalog[0]))
    return SimpleNamespace(_reconstruction_stack=lambda: trace,
        languageSpace=language, inputSpace=SimpleNamespace(
            _word_active_mask=torch.ones(1, words, dtype=torch.bool)),
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=2)))

def test_deep_sentence_derivation_preserves_exact_postorder():
    from Models import BasicModel
    words = 600
    positions, actions, targets, columns = BasicModel._derivation_program(
        _deep_derivation_owner(words), budget=3*words)
    assert positions.tolist() == [list(range(words))]
    expected = [(0, -1, 0), (2, 0, 0)]
    addresses = [-1, 2]
    for word in range(1, words):
        expected.extend(((0, -1, word), (1, 0, -1), (2, 0, 0)))
        addresses.extend((-1, 3*word, 3*word+2))
    assert actions.tolist() == [[list(action) for action in expected]]
    assert columns.tolist() == [addresses]
    assert targets[0, :6].tolist() == [1, 0, 2, 1, 0, 2]
    assert targets[0, -3:].tolist() == [1, 2, -1]


def test_deep_meaning_projection_accepts_configured_sentences():
    from Language import LanguageSpace
    words = 1200
    rows = [(0, -1, 0)]
    for word in range(1, words):
        rows.extend(((0, -1, word), (1, 0, -1)))
    entry = SimpleNamespace(actions=torch.tensor(rows), leaves=torch.ones(words, 2),
                            concept_ids=torch.arange(1, words+1))
    owner = SimpleNamespace(_compose_binary_rules=[SimpleNamespace(method_name='sum')],
                            _compose_unary_rules=[])
    registry = SimpleNamespace(form=lambda *a, **k: None)
    # A sum-only selected program makes no grammatical claim. Recovery is
    # bounded, but projecting its structure must accept the configured length.
    assert LanguageSpace.program_meaning(owner, entry, registry) is None


def test_program_keeps_the_identity_leaf_stamp_before_reference_resolution():
    from chunk_fixtures import Walk,choice
    from MediumTermMemory import MediumTermMemory
    walk=Walk((SimpleNamespace(method_name='sum',head_role=0),),())
    state=walk.empty(torch.ones(1,4),3)
    state=walk.shift(state,torch.ones(1,4),word=101,index=0,order=1)
    state=walk.shift(state,torch.ones(1,4),word=102,index=1,order=2)
    state=walk.apply(state,choice(torch.ones(1,4),head=0,order=2),torch.ones(1,2,4),2)
    trace=SimpleNamespace(_choice_values=state[3],_chunk_state=state[0],
        _chunk_scope=state[1],_choice_transactions=state[2])
    observed=[]
    def resolve(owner,rows,ids,actions,*,leaf_orders,admit):
        observed.append(leaf_orders.clone())
        return ids.clone(),torch.full_like(ids,-1)
    model=SimpleNamespace(_reconstruction_stack=lambda:trace,inputSpace=SimpleNamespace(),
        symbolSpace=SimpleNamespace(_word_reference_orders=torch.tensor([[0,2]])),
        languageSpace=SimpleNamespace(resolve_lexical_references=resolve),
        _concept_owner=lambda:SimpleNamespace(mtm=MediumTermMemory(n_vectors=512)))
    program=(torch.tensor([[0,1]]),torch.tensor([[[0,-1,0],[0,-1,1],[1,0,-1]]]),
             torch.tensor([[0,0,0]]),torch.tensor([[0,1,2]]))
    entry,=BasicModel._program_entries(model,program,torch.zeros(1,2,4),
        torch.tensor([[10,11]]),torch.tensor([[10,11]]),torch.ones(1,2),
        torch.zeros(1,3,4),concept_ids=torch.tensor([[101,102]]))
    assert entry.leaf_orders.tolist()==[1,2]
    assert observed[0].tolist()==[1,2]
    assert entry.identity_leaves.tolist()==[1,0]
    assert entry.identity_ids.tolist()==[512]
    assert entry.concept_ids.tolist()==[101,102]
    assert model.symbolSpace._word_reference_orders.tolist()==[[0,2]]



def test_identity_admission_uses_the_resolved_stm_order(tmp_path, monkeypatch, eager_reading):
    from Models import FunctionalPeerSTM
    import ChunkJournal
    import STMChunks
    from test_compiled_word_chunk import _tiny_canonical_model
    from What import What
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        concept_rows=128, input_width=16, batch_size=1, chooser_depth=1,
        training_overrides={'reconstructionPlacement': 'eager'},
        architecture_overrides={'answerSynthesis': False})
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    resolve, push = FunctionalPeerSTM.resolve_top_reference, ChunkJournal.push
    expected, observed = [], []
    def resolved(*args, **kwargs):
        result = resolve(*args, **kwargs)
        expected.append(STMChunks.stack_view(result)[2][:, 0].clone())
        return result
    def opening(state, scope, active, **kwargs):
        if active.any():
            torch.testing.assert_close(STMChunks.stack_view(state)[2][:,0][active], expected[-1][active], rtol=0, atol=0)
            observed.append(True)
        return push(state, scope, active, **kwargs)
    monkeypatch.setattr(FunctionalPeerSTM, 'resolve_top_reference', staticmethod(resolved))
    monkeypatch.setattr(ChunkJournal, 'push', opening)
    try:
        raw = model.inputSpace.prepInput(['a b'])
        model.runBatch(train=False, batchSize=1, split='validation',
            batch_override=(raw, torch.empty(1, 0)), questions=(What.present(0, split='validation'),))
        assert observed
    finally:
        model.End()


def test_unresolved_typed_admission_keeps_the_scored_source_reference():
    from Language import LanguageSpace
    rule = SimpleNamespace(method_name='kind', head_role=1, determiner_mode='kind',
        reference_roles=('argin1',), reference_modes=(), order_delta=0)
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    owner = SimpleNamespace(resolve_word_concept=lambda *a, **k: None,
        word_surface_for_row=lambda row: 'pending-form', word_concepts=lambda surface: (101,),
        _concept_source_order=lambda identity: 0,
        definitions=SimpleNamespace(word=lambda **kwargs: 101),
        interpret=SimpleNamespace(forward=lambda *a, **k: None))
    arguments = (owner, torch.tensor([10]), torch.tensor([101]),
                 torch.tensor([[0, -1, 0], [2, 0, 0]]))
    results = [LanguageSpace.resolve_lexical_references(language, *arguments,
        leaf_orders=torch.tensor([0]), admit=admit) for admit in (False, True)]
    assert results[0][0].tolist() == results[1][0].tolist() == [101]
    assert results[0][1].tolist() == results[1][1].tolist() == [2]


def test_ambiguous_typed_reading_is_open_in_preview_and_admission():
    from Language import LanguageSpace
    rule = SimpleNamespace(method_name='kind', head_role=1, determiner_mode='kind',
        reference_roles=('argin1',), reference_modes=(), order_delta=0)
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    owner = SimpleNamespace(resolve_word_concept=lambda *a, **k: None,
        word_surface_for_row=lambda row: 'pending-form', word_concepts=lambda surface: (201, 202),
        _concept_source_order=lambda identity: 2,
        definitions=SimpleNamespace(word=lambda **kwargs: 101),
        interpret=SimpleNamespace(forward=lambda *a, **k: None))
    arguments = (owner, torch.tensor([10]), torch.tensor([101]),
                 torch.tensor([[0, -1, 0], [2, 0, 0]]))
    results = [LanguageSpace.resolve_lexical_references(language, *arguments,
        leaf_orders=torch.tensor([0]), admit=admit) for admit in (False, True)]
    assert results[0][0].tolist() == results[1][0].tolist() == [-1]
    assert results[0][1].tolist() == results[1][1].tolist() == [2]
