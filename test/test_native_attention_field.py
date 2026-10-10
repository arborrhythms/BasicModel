"""Native support reads, independent meters and sentence lifetime isolation."""
import torch


def _model(tmp_path, monkeypatch):
    import util
    from test_compiled_word_chunk import _tiny_canonical_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'intraLossWeight': 0., 'reconstructionPlacement': 'eager'})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def test_native_reads_one_supported_word_and_charges_only_kept_trial(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    model.candidate_attention = None  # the specified source-order control
    try:
        inputs = model.inputSpace.prepInput(['alpha alpha', 'beta beta'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        report = model._last_sentence_field
        assert len(report['trials']) == 2
        for trial in report['trials']:
            assert (trial['when'][:, 0, 1] <= trial['when'][:, 1, 0]).all()
            assert trial['admission'].bool().all()
            assert len(trial['reads']) == 2
            assert all(read['admitted'].sum(-1).eq(1).all() for read in trial['reads'])
        assert report['iterations'].tolist() == [2, 2]
        assert [m.counts['serial-word-loop'] for m in model._field_meters] == [2, 2]
        assert all('serial-word-loop' not in m.counts for m in model._attention_meters)
        assert all(torch.isfinite(value).all() for value in model._last_field_cost.values())
        torch.testing.assert_close(model._last_field_cost['work'], torch.full((2,), 2 * model.WHAT_STEP_COST))
        assert model._sentence_field is None and model._sentence_read_order is None
        # The NULL closing must not erase the final read's symbol activation.
        record = model._last_sentence_understanding
        assert record.word_values[record.word_valid].abs().sum(-1).gt(0).all()
        terms = model._sentence_cost_registry._terms
        assert not terms['reconstruction.free_bytes']['trained']
        assert terms['reconstruction.field']['trained']
        assert terms['reconstruction.iterations']['trained']
    finally:
        model.End()


def test_new_run_resets_trial_scratch_even_if_the_body_raises(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    try:
        model._sentence_field = object()
        model._last_sentence_field = object()
        model._field_meters = (object(),)
        try:
            with model._sentence_run():
                assert model._sentence_field is None and model._last_sentence_field is None
                assert model._field_meters == ()
                model._sentence_field = object()
                model._sentence_read_order = ([1, 0],)
                raise RuntimeError('fixture abort')
        except RuntimeError as error:
            assert str(error) == 'fixture abort'
        assert model._sentence_field is None and model._sentence_read_order is None
    finally:
        model.End()


def test_trial_leaf_mask_does_not_leak_to_the_next_trial_or_commit(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    from SentenceAttention import SentenceField
    original = model._sentence_path_cost
    masks = []
    def cost(*args):
        masks.append(model.inputSpace._ar_grammar_leaf_mask.clone())
        result = original(*args)
        # Perturb scratch only after this trial's complete observation. The
        # following trial and admission must use their own native source mask.
        model.inputSpace._ar_grammar_leaf_mask = torch.zeros_like(masks[-1])
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', cost)
    model.candidate_attention = None
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        assert len(masks) == 2 and masks[0].any()
        torch.testing.assert_close(masks[0], masks[1])
        assert model.inputSpace._ar_grammar_leaf_mask.any()
    finally:
        model.End()


def test_native_candidate_departure_credits_only_scorer_parameters(tmp_path, monkeypatch):
    import SentenceCredit
    model = _model(tmp_path, monkeypatch)
    original_draw = SentenceCredit.departure
    def candidate_draw(*args, **kwargs):
        draw = original_draw(*args, **kwargs)
        eligible = kwargs['candidates']
        rows = eligible.any(-1)
        draw.update(walk=torch.where(rows, 2, draw['walk']),
            candidate_round=torch.where(rows, eligible.long().argmax(-1), -1),
            narrowing=torch.zeros_like(rows), attention_round=torch.full_like(draw['walk'], -1),
            compose_round=torch.full_like(draw['walk'], -1),
            walk_count=torch.ones_like(draw['walk_count']), walk_rounds=eligible.sum(-1))
        return draw
    monkeypatch.setattr(SentenceCredit, 'departure', candidate_draw)
    original_cost = model._sentence_path_cost
    def fixed_comparison(*args):
        result = original_cost(*args)
        registry = model._sentence_cost_registry
        original_total = registry.total
        value = 1. if model._sentence_trial == 'explore' else 2.
        def total(*args, **kwargs):
            cost = original_total(*args, **kwargs)
            return cost - cost.detach() + value if kwargs.get('objective') == 'reconstruction' else cost
        registry.total = total
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', fixed_comparison)
    before = {key: value.clone() for key, value in model.candidate_attention.state_dict().items()}
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
        optimizer = model.getOptimizer(lr=.003)
        model.runBatch(train=True, optimizer=optimizer, batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        report = model._last_sentence_field
        assert report['backup']['rows'].all() and report['wins'].all()
        assert report['trials'][0]['reads'][0]['action'].tolist() == [0, 0]
        assert report['trials'][1]['reads'][0]['action'].tolist() == [1, 1]
        assert any(not torch.equal(value, before[key]) for key,value in model.candidate_attention.state_dict().items())
        gradient = model.ownership_gradient_diagnostics(optimizer)
        parameters = [row for row in gradient['parameters'] if row['parameter'].startswith('candidate_attention.')]
        assert parameters and any(row['writers'] == ['attention'] for row in parameters)
        assert all(not row['writers'] or row['writers'] == ['attention'] for row in parameters)
    finally:
        model.End()


def test_rereading_cannot_bind_the_occurrence_or_embedded_rows_it_replaces():
    from ReferenceContext import exclude_replaced_occurrences
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    from Occurrence import document_digest, sentence_key
    store = TernaryTruthStore(4, capacity=8)
    point = ConceptualMeaning.from_description(torch.ones(4))
    content = sentence_key((b'field',))
    rows = []
    for document, position in (('current', 1), ('current', 2), ('other', 1),
                               (('embedded', document_digest('current'), 1), 0)):
        rows.append(store.append_meaning(point, document_key=document,
                                        sentence_index=position, content_key=content))
    ids = store.row_ids[rows]
    valid = torch.ones(len(rows), dtype=torch.bool)
    result = exclude_replaced_occurrences(store, ids, valid, 'current', 1)
    assert result.tolist() == [False, True, True, False]
    assert valid.all() and len(store) == 4


def test_decomposition_targets_follow_actual_reversed_candidate_reads(tmp_path, monkeypatch):
    from test_reverse_traversal import _select_completed_binary_path
    model = _model(tmp_path, monkeypatch)
    _select_completed_binary_path(model)
    monkeypatch.setattr(model.candidate_attention, 'forward',
        lambda context, candidates: torch.arange(candidates.shape[1], device=candidates.device,
            dtype=candidates.dtype)[None].expand(candidates.shape[:2]))
    original = model._decomposition_teacher_loss
    checked = []
    def teacher(observation, record):
        result = original(observation, record)
        for report in model._last_decomposition_teacher:
            if 'target' in report and min(report['target']) >= 0:
                row = report['row']
                # The fixture reads source word 1, then source word 0. These
                # are native symbol identities, not an inferred nearest code.
                assert report['target'] == record.word_rows[row, [1, 0]].tolist()
                checked.append(report)
        return result
    monkeypatch.setattr(model, '_decomposition_teacher_loss', teacher)
    # Isolate this read-order certificate from an independent reorder trial.
    import SentenceCredit
    departure = SentenceCredit.departure
    monkeypatch.setattr(SentenceCredit, 'departure',
        lambda *args, **kwargs: departure(*args, **(kwargs | {'candidates': None})))
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        assert checked
        assert model._last_sentence_field['trials'][0]['reads'][0]['action'].tolist() == [1, 1]
    finally:
        model.End()
