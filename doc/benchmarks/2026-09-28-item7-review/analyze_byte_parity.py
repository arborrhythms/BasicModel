"""Locate the first differing byte-score operation on captured native inputs."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]


def main():
    import torch
    import Models
    torch.set_num_threads(1)
    folder = HERE / 'byte-diagnostic'
    reports = {}
    for mode in ('packed', 'single'):
        a = json.loads((folder / (mode + '.json')).read_text())['parity']
        b = json.loads((HERE / 'measurements' / (mode + '.json')).read_text())['parity']
        assert a == b, 'Python diagnostic differs from the primary native receipt'
    source = textwrap.dedent(inspect.getsource(Models.BasicModel._byte_word_cost))
    final = 'return (-logp * pos).sum(-1) / pos.sum(-1).clamp_min(1.0)'
    assert final in source
    scope = dict(vars(Models))
    exec(compile(source.replace(final, 'return locals()'), '<byte-score observer>', 'exec'), scope)
    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=Models.BasicModel._BYTE_ASSIGNMENT_TAU)
    captured = {mode: torch.load(folder / (mode + '.scorer.pt'), weights_only=True)
                for mode in ('packed', 'single')}
    # Both lists are in reverse traversal order. Index 7 is the final "1"
    # of the fourth sentence, the only per-word byte-cost discrepancy.
    inputs = {mode: dict(calls[7]) for mode, calls in captured.items()}
    traces = {}
    for mode, values in inputs.items():
        values.pop('cost')
        traces[mode] = scope['_byte_word_cost'](owner, **values)
    a, b = traces['packed'], traces['single']
    row = 1
    ia, ib = (t['present'][row].nonzero().flatten() for t in (a, b))
    report = dict(primary_receipt_reproduced_exactly=True,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        sentence=3, word='1', packed_word_column=int(inputs['packed']['word']),
        single_word_column=int(inputs['single']['word']),
        packed_candidate_columns=ia.tolist(), single_candidate_columns=ib.tolist(),
        equal_recovered_idea=torch.equal(a['idea'][row], b['idea'][row]),
        equal_normalized_candidates=torch.equal(a['bank_n'][row, ia], b['bank_n'][row, ib]),
        equal_candidate_logits=torch.equal(a['sim'][row, ia], b['sim'][row, ib]),
        equal_assignments=torch.equal(a['assign'][row, ia], b['assign'][row, ib]),
        max_assignment_difference=float((a['assign'][row, ia]-b['assign'][row, ib]).abs().max()),
        null_probabilities=[float(t['p_null'][row]) for t in (a,b)],
        costs=[float(captured[mode][7]['cost'][row]) for mode in ('packed','single')])
    # Keep all candidate values and masks; move only their columns so the
    # ordered softmax input is the single presentation's input.
    moved = dict(inputs['packed'])
    shift = int(ib[0] - ia[0])
    for key in ('bank_n','bank_bytes','bank_valid'):
        moved[key] = moved[key].clone()
        moved[key][row] = torch.roll(moved[key][row], shift, 0)
    same = scope['_byte_word_cost'](owner, **moved)
    report['column_shift_only'] = dict(shift=shift,
        equal_logits=torch.equal(same['sim'][row],b['sim'][row]),
        equal_assignments=torch.equal(same['assign'][row],b['assign'][row]),
        equal_byte_probabilities=torch.equal(same['pred'][row],b['pred'][row]),
        cost=float(Models.BasicModel._byte_word_cost(owner,**moved)[row]))
    (folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
