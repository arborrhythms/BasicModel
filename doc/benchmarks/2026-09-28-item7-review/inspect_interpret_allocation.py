"""Trace the structural allocation change behind the interpret counterfactual.

This is a small mechanism diagnostic, not a replacement baseline. Each side
starts from the same seed and codebook; only InterpretLayer.forward changes.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test'), str(HERE)]


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    import torch
    import Interpret
    from bisect_components import historical_method
    from Spaces import _concept_alloc_of
    from test_cs_sparse_weights import _cs
    torch.set_num_threads(1)
    current = Interpret.InterpretLayer.forward
    namespace = dict(vars(Interpret))
    code = historical_method('bin/Interpret.py', 'forward')
    exec(compile(code, '<HEAD InterpretLayer.forward>', 'exec'), namespace)
    results = {}
    try:
        for name, forward in (('current', current), ('head', namespace['forward'])):
            Interpret.InterpretLayer.forward = forward
            torch.manual_seed(42)
            cs = _cs(nS=256, order=3)
            interpret = Interpret.InterpretLayer(conceptualSpace=cs)
            alloc = _concept_alloc_of(cs)
            basis = cs.similarity_codebook.getW()
            report = dict(initial_dictionary=digest(basis), words=[])
            for spelling, part in (('nine', 7), ('plus', 8), ('one', 9)):
                rng = digest(torch.get_rng_state())
                word = interpret.lookup_word([part], [], form=spelling)
                obj = interpret.forward(word, order=1)
                meta = alloc.interpretations[(word, 1)][1]
                object_row = cs._csw_row_of(obj)
                report['words'].append(dict(spelling=spelling, word=word, object=obj,
                    meta=meta, next_id=alloc.next_id,
                    object_row=object_row, object_code=digest(basis[object_row]),
                    meta_row=cs._csw_row_of(meta),
                    meta_order=cs._concept_source_order(meta),
                    meta_parts=cs.concept_parts(meta), meta_wholes=cs.concept_wholes(meta),
                    rng_unchanged=rng == digest(torch.get_rng_state())))
            results[name] = report
    finally:
        Interpret.InterpretLayer.forward = current
    output = HERE / 'bisect/interpret-allocation.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
