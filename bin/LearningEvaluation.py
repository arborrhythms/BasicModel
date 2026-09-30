"""Evidence of training exposure, separate from evidence of learned quality.

Only successful, addressed FineWeb training presentations count. Reading,
validation, corpus size and optimizer-step estimates do not qualify a model.
These host counters never inspect accelerator tensor values.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from functools import wraps
from numbers import Integral

FINEWEB_CORPUS = 'karpathy/fineweb-edu-100b-shuffle'
MIN_FINEWEB_SENTENCES = 1_000_000


def scaler_step_performed(scaler, optimizer):
    """Observe AMP's host-side skip without synchronizing its device flag.

    A fused optimizer may perform its overflow check inside step(). Its
    update cannot be certified here, so conservatively do not count it.
    """
    previous = optimizer.step
    had_override = 'step' in vars(optimizer)
    override = vars(optimizer).get('step')
    performed = False

    @wraps(previous)
    def observed(*args, **kwargs):
        nonlocal performed
        result = previous(*args, **kwargs)
        performed = True
        return result

    optimizer.step = observed
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        if had_override:
            optimizer.step = override
        else:
            del optimizer.step
    return performed and not getattr(optimizer, '_step_supports_amp_scaling', False)


def validated_progress(value):
    if value is None:
        return None  # Historical checkpoints have unknown exposure.
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('invalid FineWeb training progress version')
    for name in ('sentences', 'updates'):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f'FineWeb training {name} must be a nonnegative integer')
    if not isinstance(value.get('manifests'), dict):
        raise ValueError('FineWeb training progress requires source manifests')
    return copy.deepcopy(value)


def record_fineweb_training(model, *, split, source_rows):
    """Called once after the main optimizer step, excluding exploration.

    Count completed fields, including ragged packed rows, against
    the source addresses of this batch. This is presentations, not unique
    sentences: repeated training epochs count, inference does not.
    """
    data = getattr(getattr(model, 'inputSpace', None), 'data', None)
    manifest = getattr(data, 'source_manifest', None)
    if (split != 'train' or getattr(model, '_preflight_active', False)
            or not isinstance(manifest, dict)
            or manifest.get('dataset') != 'text'
            or manifest.get('corpus') != FINEWEB_CORPUS
            or not manifest.get('shards') or source_rows is None):
        return
    addresses = getattr(data, 'source_addresses', {}).get('train', ())
    understanding = getattr(model, '_last_understanding', None)
    programs = getattr(understanding, 'sentence_fields', {})
    if not programs:
        programs = {0: getattr(understanding, 'sentence_states', ())}
    count = 0
    for slot, rows in programs.items():
        for row, program in enumerate(rows):
            if program is None or row >= len(source_rows):
                continue
            source = source_rows[row]
            if isinstance(source, (tuple, list)):
                source = source[slot] if 0 <= slot < len(source) else None
            elif slot != 0:
                continue
            if (isinstance(source, Integral) and not isinstance(source, bool)
                    and 0 <= source < len(addresses)
                    and addresses[source].get('split', 'train') == 'train'):
                count += 1
    if not count:
        return
    progress = getattr(model, '_fineweb_training_progress', None)
    if progress is None:
        progress = dict(version=1, corpus=FINEWEB_CORPUS, sentences=0,
                        updates=0, manifests={})
        model._fineweb_training_progress = progress
    progress['sentences'] += count
    progress['updates'] += 1
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if key not in progress['manifests']:
        progress['manifests'][key] = copy.deepcopy(manifest)


@dataclass(frozen=True)
class LearningReadiness:
    eligible: bool
    sentences: int
    required: int
    reason: str


def fineweb_readiness(training_state, *, minimum_sentences=MIN_FINEWEB_SENTENCES):
    if type(minimum_sentences) is not int or minimum_sentences < 1:
        raise ValueError('minimum FineWeb sentences must be a positive integer')
    progress = validated_progress(training_state.get('fineweb_training_progress'))
    if progress is None:
        return LearningReadiness(False, 0, minimum_sentences,
            f'FineWeb exposure is unrecorded; requires {minimum_sentences:,} completed training sentences')
    count = progress['sentences']
    known = (progress.get('corpus') == FINEWEB_CORPUS and progress['updates'] > 0
             and bool(progress['manifests']) and all(
                 isinstance(m, dict) and m.get('corpus') == FINEWEB_CORPUS
                 for m in progress['manifests'].values()))
    ready = known and count >= minimum_sentences
    return LearningReadiness(ready, count, minimum_sentences,
        f'{count:,}/{minimum_sentences:,} completed FineWeb training sentences'
        + ('' if known else '; FineWeb source provenance is unavailable'))


def checkpoint_readiness(path, *, minimum_sentences=MIN_FINEWEB_SENTENCES):
    """Read metadata before constructing a large model; corruption is an error."""
    import torch
    bundle = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(bundle, dict):
        raise ValueError('learning evaluation requires an integrated checkpoint')
    return fineweb_readiness(bundle.get('training_state', {}),
                             minimum_sentences=minimum_sentences)
