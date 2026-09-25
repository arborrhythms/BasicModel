"""Repeat the historical workload, recording the inverse's actual sealed inputs.

Seed 42 is a declared measurement, not a test seed. No optimizer updates are
made in parity mode. The old workload owns vocabulary warmup and reset order.
"""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '2026-09-21-item10'
sys.path.insert(0, str(PREVIOUS))
spec = importlib.util.spec_from_file_location('historical_probe', PREVIOUS / 'probe.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
import parity
import torch

original_measure = parity.measure


def measure(model, mode):
    reconstruct, run_batch = model._reconstruct_sentences, model.runBatch
    boundary, batches = {}, []

    def capture_reconstruction(S, reference, roots, depth=None, end=None, end_depth=None):
        boundary.update(root=roots.detach().clone(), end=end.detach().clone(),
                        depth=depth.detach().clone(), end_depth=end_depth.detach().clone())
        return reconstruct(S, reference, roots, depth, end, end_depth)

    def capture_batch(*args, **kwargs):
        result = run_batch(*args, **kwargs)
        isp = model.inputSpace
        active, ids = isp._word_active_mask, isp._packed_sentence_ids
        bank_rows = isp._ar_concept_lookup_rows
        bank_sentence = getattr(isp, '_ar_concept_lookup_sentence_ids', None)
        records = []
        for b in range(active.shape[0]):
            present = ids[b, active[b]].unique().tolist()
            for sid in present:
                mask = active[b] & (ids[b] == sid)
                final = sid == max(present)
                root = boundary['end'][b] if final else boundary['root'][b, sid].reshape(3, -1)
                record = dict(row=b, slot=sid, root=root.reshape(-1).tolist(),
                              depth=int(boundary['end_depth'][b] if final else boundary['depth'][b, sid]),
                              candidate_rows=bank_rows[b, bank_rows[b] >= 0].tolist())
                scope = bank_rows[b] >= 0
                if torch.is_tensor(bank_sentence):
                    scope = scope & (bank_sentence[b] == sid)
                record['effective_candidate_rows'] = bank_rows[b, scope].tolist()
                record['inputs'] = {}
                for name in ('_ar_word_part_ids', '_ar_word_part_mask', '_ar_word_part_offsets',
                             '_ar_word_concept_rows', '_ar_word_object_rows', '_ar_word_concept_orders',
                             '_ar_word_object_orders', '_ar_readout_coefficients', '_ar_percept_reference_codes',
                             '_ar_percept_reference_roles', '_ar_target_word_bytes', '_ar_target_word_mask'):
                    value = getattr(isp, name)
                    record['inputs'][name] = value[b, mask].detach().tolist()
                program = model._last_understanding.sentence_programs[int(sid)][b]
                record['actions'] = program.actions.tolist()
                record['program_root'] = program.end_state.reshape(-1).tolist()
                records.append(record)
        batches.append(records)
        return result

    model._reconstruct_sentences, model.runBatch = capture_reconstruction, capture_batch
    try:
        report = original_measure(model, mode)
        records = {(r['row'], r['slot'] if mode == 'packed' else index): r
                   for index, batch in enumerate(batches[1:]) for r in batch}
        for sentence in report['sentences']:
            address = sentence['source_row']
            observed = records[address // 2, address % 2]
            sentence['program_root'] = observed.pop('program_root')
            sentence['root'] = observed.pop('root')
            sentence['reverse_inputs'] = observed
        report['root_source'] = 'actual arguments at _reconstruct_sentences; program_root is independently captured'
        return report
    finally:
        model._reconstruct_sentences, model.runBatch = reconstruct, run_batch


parity.measure = measure
if __name__ == '__main__':
    baseline.main()
