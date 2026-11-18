"""Quality probes for an explicitly supplied, sufficiently trained checkpoint.

These read held-out FineWeb without optimizer updates. The checkpoint's
recorded exposure controls eligibility; it does not predict that a gate passes.
"""
from dataclasses import replace
from pathlib import Path
import os
import torch

from LearningEvaluation import MIN_FINEWEB_SENTENCES, checkpoint_readiness


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


def read_validation(model, *, sentences=64, documents=1024):
    """Read chronological held-out inputs without forecast targets or updates."""
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
    if discourse is None:
        raise ValueError('learning evaluation requires symbol activation expectation')
    observations = []
    for index, text in enumerate(data.validation_input[:sentences]):
        doc = addresses[index]['document']
        raw = model.inputSpace.prepInput([text])
        with torch.no_grad():
            model.runBatch(train=False, split='validation', batchSize=1,
                source_rows=[index], batch_override=(raw, torch.empty(1, 0)),
                questions=(What.present(0, split='validation'),))
        if not discourse._inter_context[0]:
            raise ValueError('held-out input produced no complete prediction context')
        observations.append(dict(row=index, document=doc,
            occurrences=[frame.occurrence for frame in discourse.situation_references(0)]))
        same_doc = index + 1 < sentences and addresses[index + 1]['document'] == doc
        model.flush_word_buffers()
        model.dispatch_per_row_reset([not same_doc])
        model.dispatch_soft_reset()
        model.post_tick_compact()
    return dict(source_manifest=data.source_manifest, sentences=sentences, observations=observations)
