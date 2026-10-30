"""Measure a prepared corpus through the native loader and packed cursor.

This is a data/capacity measurement, not a training or acquisition result.
The schedule uses the configured native WholeSpace's initial tiling. It is
not a claim about a later learned tiling or checkpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from functools import lru_cache


def initial_counter(config):
    """Construct the run's actual property inventory and native packing seam."""
    from types import SimpleNamespace
    from util import init_config, TheXMLConfig, ProjectPaths
    from WordIdentity import resolve_identity_layout
    from Spaces import WholeSpace, InputSpace, _ladder_periods
    from architecture import canonical_shape
    init_config(path=str(config), defaults_path=str(Path(ProjectPaths.DATA_DIR) / 'model.xml'))
    cfg = TheXMLConfig.data
    resolve_identity_layout(cfg)
    section = cfg['WholeSpace']
    whole = WholeSpace([section['nInput'], section['nInputDim']],
                       [section['nVectors'], section['nDim'] - sum(canonical_shape('WholeSpace'))],
                       [section['nOutput'], section['nOutputDim']])
    proxy = SimpleNamespace(_unit_span_fn=whole.unit_spans_of_bytes)
    @lru_cache(maxsize=16384)
    def count(text):
        return InputSpace.sentence_unit_count(proxy, text, trailing_space=True)
    return count, dict(config=str(Path(config).resolve()),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        tiling='native initial WholeSpace, no boundary updates',
        whole_space=section,
        when_ladder_periods=_ladder_periods(cfg['SymbolSpace']['ltmCapacity']),
        byte_capacity=int(cfg['InputSpace']['nOutput']),
        root_slots=min(int(cfg['architecture']['serialWordCapacity']),
                       InputSpace.PACKED_SENTENCE_SLOTS))


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def audit(manifest, output, *, streams, word_capacity, config, max_docs=None, max_units=None):
    from data import Data
    import Meronomy
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = Data()
    data.load('curriculum', dat={'corpusManifest': str(Path(manifest).resolve()),
              'maxSentenceUnits': max_units},
              max_docs=max_docs, max_sentence_words=word_capacity)
    report = dict(manifest=data.source_manifest, streams=streams, word_capacity=word_capacity,
        load_seconds=time.monotonic() - started, peak_bytes_after_load=peak_bytes(),
        split_sentences={split: len(getattr(data, split + '_input'))
                         for split in ('train', 'validation', 'test')})
    (output / 'load.json').write_text(json.dumps(report, indent=2) + '\n')
    words, counts = {}, Counter()
    for split in ('train', 'validation', 'test'):
        for sentence in getattr(data, split + '_input'):
            raw = sentence.encode('ascii', errors='replace')
            for start, end in Meronomy.word_spans(raw):
                word = raw[start:end]
                words.setdefault(word, len(words))
                counts[word] += 1
    with (output / 'word-forms.jsonl').open('w') as target:
        for word in words:
            target.write(json.dumps(dict(form=word.decode('ascii'), count=counts[word])) + '\n')
    report['word_forms'] = len(words)
    report['word_observations'] = sum(counts.values())
    report['longest_word'] = max(map(len, words), default=0)
    report['word_form_order'] = 'first occurrence in train, validation, test; no learner seed'
    print(json.dumps({k: report[k] for k in ('load_seconds', 'word_forms', 'word_observations')}), flush=True)
    counter, counter_record = initial_counter(config)
    report['packing'] = counter_record
    cursor = data.data_loader('train', streams).dataset
    options = dict(max_bytes=counter_record['byte_capacity'],
                   max_sentences=counter_record['root_slots'])
    ticks, consumed, checksum = 0, 0, hashlib.sha256()
    stage_counts, checkpoints = Counter(), []
    previous_stage = None
    while not cursor.packed_done():
        try:
            rows, _targets, ends = cursor.next_packed_tick(word_capacity, counter, **options)
        except ValueError as error:
            report.update(packed_ticks=ticks, consumed_train_sentences=consumed,
                          status='capacity_failure', error=str(error), peak_bytes=peak_bytes(),
                          total_seconds=time.monotonic() - started)
            (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
            raise
        sources = cursor.last_source_indices
        encoded = json.dumps(sources, separators=(',', ':')).encode()
        checksum.update(encoded + b'\n')
        stage = cursor._stage_index
        if stage != previous_stage:
            checkpoints.append(dict(skip_ticks=ticks, source_rows=sources, stage_index=stage))
            previous_stage = stage
        consumed += sum(map(len, sources))
        stage_counts[stage] += sum(map(len, sources))
        ticks += 1
    expected = len(data.train_input)
    # Empty/nonword-only inputs deliberately consume no transaction in the
    # existing packed cursor. Report them explicitly rather than silently
    # treating them as witnessed sentences.
    nonword = sum(counter(text) < 1 for text in data.train_input)
    if consumed != expected - nonword:
        raise RuntimeError('curriculum cursor omitted or duplicated training sentences')
    cursor.reset_cursor()
    by_tick = {check['skip_ticks']: check for check in checkpoints}
    replay_checksum, replay_ticks = hashlib.sha256(), 0
    while not cursor.packed_done():
        cursor.next_packed_tick(word_capacity, counter, **options)
        sources = cursor.last_source_indices
        replay_checksum.update(json.dumps(sources, separators=(',', ':')).encode() + b'\n')
        if replay_ticks in by_tick and sources != by_tick[replay_ticks]['source_rows']:
            raise RuntimeError('resume_skip changed a stage boundary source address')
        replay_ticks += 1
    if replay_ticks != ticks or replay_checksum.digest() != checksum.digest():
        raise RuntimeError('resume_skip changed the ordered source sequence')
    report.update(packed_ticks=ticks, consumed_train_sentences=consumed,
        nonword_train_sentences=nonword, source_sequence_sha256=checksum.hexdigest(),
        resume_same_sequence=True, resume_stage_boundaries=checkpoints,
        stage_train_sentences=dict(stage_counts), peak_bytes=peak_bytes(),
        total_seconds=time.monotonic() - started)
    from RowTime import HORIZON, encoder
    row_encoder = encoder()
    positions = [address['sentence'] + 1 for address in data.source_addresses['train']]
    report['clock_range'] = dict(
        processing_ticks_for_this_training_pass=ticks,
        processing_time_basis='BasicModel.when_time advances once per processed brick',
        source_sentence_position_min=min(positions),
        source_sentence_position_max=max(positions),
        source_time_basis='Percept event time is the original document-relative sentence address',
        percept_ladder_periods=counter_record['when_ladder_periods'],
        row_time_horizon=HORIZON,
        row_time_ladder_periods=[row_encoder.maxVal, row_encoder.period_hf],
        native_training_executed=False)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('packed_ticks', 'resume_same_sequence', 'peak_bytes', 'total_seconds')}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--streams', type=int, default=28)
    parser.add_argument('--word-capacity', type=int, default=256)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-docs', type=int)
    parser.add_argument('--max-units', type=int)
    args = parser.parse_args()
    audit(args.manifest, args.output, streams=args.streams,
          word_capacity=args.word_capacity, config=args.config,
          max_docs=args.max_docs, max_units=args.max_units)


if __name__ == '__main__':
    main()
