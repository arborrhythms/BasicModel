"""Quality probes for an explicitly supplied, sufficiently trained checkpoint.

These read held-out FineWeb without optimizer updates. The checkpoint's
recorded exposure controls eligibility; it does not predict that a gate passes.
"""
from dataclasses import replace
from pathlib import Path
import os
import torch

from LearningEvaluation import MIN_FINEWEB_SENTENCES, checkpoint_readiness
from bench_sentence_expectation import (
    meaning_windows, _controlled_inputs, _score_head, score_prediction_thought)


def load_model(config, checkpoint, *, minimum_sentences=MIN_FINEWEB_SENTENCES):
    """Strict restore, without an implicit corpus load or a fresh-model fallback."""
    ready = checkpoint_readiness(checkpoint, minimum_sentences=minimum_sentences)
    if not ready.eligible:
        raise ValueError(ready.reason)
    from data import Data
    from Models import BaseModel
    from util import init_device, init_compile_backend
    import Language
    init_device(os.environ.get('BASICMODEL_DEVICE', 'cpu'))
    init_compile_backend('none')
    data = Data()
    data.input_presence = False
    Language.TheGrammar._configured = False
    previous = os.environ.get('BASIC_AUTOLOAD')
    os.environ['BASIC_AUTOLOAD'] = 'false'
    try:
        model, _ = BaseModel.from_config(str(config), data=data)
    finally:
        if previous is None:
            os.environ.pop('BASIC_AUTOLOAD', None)
        else:
            os.environ['BASIC_AUTOLOAD'] = previous
    try:
        if not model.load_weights(str(checkpoint), strict=True, require_match=True):
            raise ValueError('the requested trained checkpoint was not loaded')
    except Exception:
        model.End()
        raise
    model.eval()
    model.set_sigma(0)
    model.checkpoint_every_batches = 0
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.reconstruction_placement = 'eager'
    return model


def _forecast_question(model):
    """Ask the declared arma operation about the latest owned observation."""
    discourse = model.symbolSpace.expectation
    occurrences = discourse._inter_context_occurrences[0]
    if not occurrences or occurrences[-1] is None:
        raise ValueError('a prediction question requires an owned observation')
    occurrence = occurrences[-1]
    store = model.symbolSpace.ltm_store
    meaning = store.meaning_of(store._index_occurrences[occurrence])
    registry = model.grammatical_thoughts
    operation = registry.operation_spec('arma')
    return registry._form_candidate(operation, registry.descriptors['arma'],
        {operation.operand_roles[0]: (
            meaning.roles.sum(0) / meaning.role_mask.sum().sqrt(), occurrence)},
        source=replace(meaning, mode='interrogative', polarity=True))


def read_validation(model, *, sentences=64, documents=1024, gain=None):
    """Read chronological held-out inputs, optionally asking what comes next.

    A forecast is scored only after its next sentence arrives. Document ends
    have no target and do not enter accuracy/work averages. The next target
    never enters the question or controller. Each gain uses a separate restore.
    """
    from What import What
    data = model.inputSpace.data
    options = dict(model.cfg['architecture']['data'])
    data.load('text', num_shards=int(options.get('numShards', 1)), max_docs=documents,
        shard_dir=options.get('shardDir'), dat=options,
        max_sentence_words=int(options.get('maxSentenceWords', 0)) or None)
    addresses = data.source_addresses['validation']
    train_docs = {r['document'] for r in data.source_addresses['train']}
    assert train_docs.isdisjoint(r['document'] for r in addresses)
    if len(data.validation_input) < sentences:
        raise ValueError('held-out corpus is shorter than the declared evaluation')
    discourse = model.symbolSpace.expectation
    if discourse is None or discourse.expectation_scope != 'structured':
        raise ValueError('learning evaluation requires structured sentence expectation')
    if gain is not None:
        model.expectation_gain = gain
    groups, trials = {}, []
    pending, previous_doc = None, None
    for index, text in enumerate(data.validation_input[:sentences]):
        doc = addresses[index]['document']
        raw = model.inputSpace.prepInput([text])
        with torch.no_grad():
            model.runBatch(train=False, split='validation', batchSize=1,
                source_rows=[index], batch_override=(raw, torch.empty(1, 0)),
                questions=(What.present(0, split='validation'),))
        if not discourse._inter_context[0]:
            raise ValueError('held-out input produced no complete prediction context')
        _depth, roles, mask = discourse._inter_context[0][-1]
        roles, mask = roles.detach().cpu(), mask.detach().cpu()
        groups.setdefault(doc, []).append((roles, mask))
        if pending is not None and previous_doc == doc:
            trials.append(dict(row=index, document=doc,
                **score_prediction_thought(pending, roles, mask)))
        # No look-ahead content is read. The known document boundary only
        # determines whether a subsequent observation can score this question.
        same_doc = index + 1 < sentences and addresses[index + 1]['document'] == doc
        pending = None
        if gain is not None and same_doc:
            question = _forecast_question(model)
            with model._query_boundary_scope((0,)), torch.no_grad():
                pending = model.run_selected_thought(question, row=0, work_budget=64)
            model._end_finished_selected_thought_episodes()
        previous_doc = doc
        model.flush_word_buffers()
        model.dispatch_per_row_reset([not same_doc])
        model.dispatch_soft_reset()
        model.post_tick_compact()
    rows = list(groups.values())
    views = meaning_windows([torch.stack([v for v, _ in group]) for group in rows],
        discourse._inter_chain_window,
        [torch.stack([m for _, m in group]) for group in rows])
    if views is None:
        raise ValueError('held-out corpus produced no within-document prediction pairs')
    scores = {}
    for seed in (0, 1, 2):
        scores[seed] = {}
        for control in ('ordered', 'shuffled', 'context_free'):
            x, mask = _controlled_inputs(views[0], views[1], control, seed)
            scores[seed][control] = _score_head(discourse._inter_predictor,
                x, mask, views[2], target_masks=views[4])
    return dict(source_manifest=data.source_manifest, sentences=sentences,
                predicted_targets=len(views[2]), controls=scores, thought=trials)
