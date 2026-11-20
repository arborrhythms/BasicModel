"""Native support reads, independent meters and sentence lifetime isolation."""
from Memory import STM0
import torch


def _model(tmp_path, monkeypatch):
    import util
    from test_compiled_word_chunk import _tiny_canonical_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def test_native_reads_one_supported_word_and_charges_only_kept_trial(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    registries = []
    path_cost = model._sentence_path_cost
    def observed(*args, **kwargs):
        result = path_cost(*args, **kwargs)
        registries.append(model._sentence_cost_registry)
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', observed)
    try:
        # MTM nominates a form on recurrence (sigma-pi §21, rule 20).
        from nomination_fixtures import nominated_forms
        nominated_forms(model, ['alpha alpha', 'beta beta'])
        inputs = model.inputSpace.prepInput(['alpha alpha', 'beta beta'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        report = model._last_sentence_field
        assert len(report['trials']) == 2
        for trial in report['trials']:
            assert (trial['when'][:, 0, 1] <= trial['when'][:, 1, 0]).all()
            assert trial['admission'].bool().all()
            assert len(trial['reads']) == 2
            assert all(read['rows'].all() for read in trial['reads'])
        assert report['iterations'].tolist() == [2, 2]
        assert all(m.counts['bracket'] > 0 for m in model._attention_meters)
        assert all('serial-word-loop' not in m.counts for m in model._attention_meters)
        assert model._last_field_cost is None  # the compose-forest reconstruction is retired
        assert model._sentence_field is None and model._sentence_read_order is None
        # The NULL closing must not erase the final read's symbol activation.
        record = model._last_sentence_understanding
        assert record.word_values[record.word_valid].abs().sum(-1).gt(0).all()
        terms = registries[-1]._terms
        assert not terms['reconstruction.free_bytes']['trained']
        assert terms['reconstruction_reverse.what']['trained']
        assert terms['reconstruction_order_zero.what']['trained']
    finally:
        model.End()


def test_new_run_resets_trial_scratch_even_if_the_body_raises(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    try:
        model._sentence_field = object()
        model._last_sentence_field = object()
        try:
            with model._sentence_run():
                assert model._sentence_field is None and model._last_sentence_field is None
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
    import SentenceCredit
    departure = SentenceCredit.departure
    def compose_departure(_narrowing, compose, **options):
        # Whole paths may depart after composition. This contract explicitly
        # exercises a second compose path so both leaf masks are observed.
        return departure(None, compose, active=options['active'],
            compose_round=options['compose_round'], used=options['used'])
    monkeypatch.setattr(SentenceCredit, 'departure', compose_departure)
    original = model._sentence_path_cost
    masks = []
    def cost(*args, **kwargs):
        masks.append(model.inputSpace._ar_grammar_leaf_mask.clone())
        result = original(*args)
        # Perturb scratch only after this trial's complete observation. The
        # following trial and admission must use their own native source mask.
        model.inputSpace._ar_grammar_leaf_mask = torch.zeros_like(masks[-1])
        return result
    monkeypatch.setattr(model, '_sentence_path_cost', cost)
    from reading_fixtures import read_in_source_order
    read_in_source_order(model, monkeypatch)
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        assert len(masks) == 2 and masks[0].any()
        torch.testing.assert_close(masks[0], masks[1])
        assert model.inputSpace._ar_grammar_leaf_mask.any()
    finally:
        model.End()


def test_reversed_candidate_reads_keep_native_field_order_without_teacher(tmp_path, monkeypatch):
    from test_reverse_traversal import _select_completed_binary_path
    model = _model(tmp_path, monkeypatch)
    _select_completed_binary_path(model)
    chooser = model._stm_reducer()
    chooser.interpret_bias = 0.  # This reading deliberately declines the first offer.
    attend, interpret = chooser.attend, chooser.choose_interpret
    def last_focus(keys, legal, space, **options):
        focus = legal[...,:2].any(-1)
        slot = torch.where(focus, torch.arange(focus.shape[1])[None], -1).amax(-1).clamp_min(0)
        options['replay_action'] = slot*6 + legal[torch.arange(len(slot)),slot,:2].long().argmax(-1)
        return attend(keys, legal, space, **options)
    chooser.trial_choices = []
    def read(word, allowed, **options):
        single = allowed.any().item()
        if single:
            chooser.trial_choices.append(None)
        options['replay_action'] = (allowed & (len(chooser.trial_choices) > 1)).long()
        return interpret(word, allowed, **options)
    monkeypatch.setattr(chooser,'attend',last_focus)
    monkeypatch.setattr(chooser,'choose_interpret',read)
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'beta alpha'])
        model.runBatch(train=True, optimizer=model.getOptimizer(lr=.003), batchSize=2,
                       batch_override=(inputs, torch.empty(2, 0)))
        assert int(model._last_sentence_field['trials'][0]['reads'][0]['index']) == 1
    finally:
        model.End()
