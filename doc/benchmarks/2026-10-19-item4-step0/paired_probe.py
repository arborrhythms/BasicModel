"""Replay the archived failing initialization; inspect, never select a seed."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import util

util.TheCompileBackend = 'none'
torch.set_num_threads(1)

import MeaningCodes
from Models import BasicModel
from test_mm_xor import _fresh_model


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
rows, current = [], {}
lane_reads = []
original_cost = BasicModel._sentence_path_cost
original_lanes = MeaningCodes.recover_lanes


def lanes(name, parent, left, right, form_width):
    result = original_lanes(name, parent, left, right, form_width)
    for side, candidate in enumerate((left, right)):
        positive, negative = candidate[..., form_width:].chunk(2, -1)
        affected = (~positive.any(-1)) & negative.any(-1)
        for row in affected.nonzero().flatten().tolist():
            lane_reads.append(plain(dict(name=name, row=row, side=side,
                trial=globals().get('model', None)._sentence_trial if 'model' in globals() else None,
                parent=parent[row], left=left[row], right=right[row],
                returned=[value[row] for value in result], form_width=form_width)))
    return result


def cost(model, *args, **kwargs):
    result = original_cost(model, *args, **kwargs)
    _, reconstruction, observation, _ = result
    record = observation['record']
    start = record.primed.meaning_start
    current.clear()
    current.update(trial=model._sentence_trial,
        root=record.root, end_depth=record.end_depth, word_valid=record.word_valid,
        word_rows=record.word_rows, words=record.word_values,
        returned=reconstruction[0], meaning_start=start,
        bank_codes=record.primed.codes, bank_rows=record.primed.rows,
        bank_weights=record.primed.weights, bank_object=id(record.primed),
        decoder=model._last_decoder_trace, decoder_comparison=model._last_decoder_comparison,
        actions=[None if p is None else p.actions for p in observation['entries']],
        frames=[None if p is None else p.operation_values for p in observation['entries']],
        reconstruction_terms={name: value * weight
            for name, value, weight, _, category in model._sentence_cost_registry.terms()
            if category == 'reconstruction'})
    rows.append(plain(dict(current)))
    return result


BasicModel._sentence_path_cost = cost
MeaningCodes.recover_lanes = lanes
model, _, _ = _fresh_model('data/XOR_grammar.xml')
report = dict(seed=1, backend='none', epochs=[], trials=rows, lane_reads=lane_reads)
try:
    optimizer = model.getOptimizer(lr=.01)
    for epoch in range(3):
        model.runEpoch(optimizer=optimizer, batchSize=4, split='train', max_batches=1)
        report['epochs'].append(plain([audit['components'] for audit in model._sentence_credit_audits]))
finally:
    model.End()
    model.symbolSpace.soft_reset()
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
