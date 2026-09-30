"""Refine the native interpret ablation to association construction alone.

Python recurrence bodies make this diagnostic inexpensive. Current and HEAD
controls are compared with the native measurements before drawing conclusions.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'), str(HERE)]


def main(mode, output):
    os.environ['MODEL_COMPILE'] = 'none'
    import torch
    import Interpret
    import measure
    from bisect_components import historical_method
    from Spaces import _concept_alloc_of
    def eager_loop(cond, body, carried_inputs):
        while bool(cond(*carried_inputs)):
            carried_inputs = body(*carried_inputs)
        return carried_inputs
    torch.while_loop = eager_loop
    source = textwrap.dedent(inspect.getsource(Interpret.InterpretLayer.forward))
    if mode == 'head':
        source = historical_method('bin/Interpret.py', 'forward')
        source = source.replace('occurrence=None,', 'occurrence=None, selected=None,')
    elif mode == 'association-head':
        line = '        cs.bind_meta((word,), (obj,), meta=meta)'
        assert line in source
        source = source.replace(line,
            "        alloc.store_of(meta).embed_pair(meta, whole_ref=('sym', word), part_ref=('sym', obj))\n"
            '        alloc.settle(meta)')
    scope = dict(vars(Interpret))
    exec(compile(source, str(output.with_suffix('.override.py')), 'exec'), scope)
    replacement = scope['forward']
    observations = []
    def observed(self, word, **kwargs):
        result = replacement(self, word, **kwargs)
        if not torch.is_tensor(word):
            cs = self.conceptualSpace
            alloc = _concept_alloc_of(cs)
            address = cs._csw_row_of(result)
            point = cs.similarity_codebook.getW()[address]
            observations.append(dict(word=int(word), requested_order=kwargs.get('order'),
                object=int(result), object_order=cs._concept_source_order(result),
                row=address, next_id=alloc.next_id,
                code_sha256=hashlib.sha256(point.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
        return result
    Interpret.InterpretLayer.forward = observed
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.override.py').write_text(source)
    try:
        measure.baseline(output)
    finally:
        output.with_suffix('.observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    report = json.loads(output.read_text())
    report['backend'] = 'diagnostic: Python recurrence bodies, MODEL_COMPILE=none'
    report['counterfactual'] = mode
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('current','head','association-head'))
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    main(args.mode,args.out)
