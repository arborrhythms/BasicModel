"""Capture actual derivation inverse mismatches without changing their result."""
import json
import os
from pathlib import Path


def pytest_configure(config):
    import DerivationReconstruction as reader
    reconstruct = reader.reconstruct

    def capture(language, record, programs, positions):
        result = reconstruct(language, record, programs, positions)
        recovered, unavailable, defined = result
        if (recovered-record.word_values).abs().max() > 1e-4:
            data = dict(recovered=recovered.tolist(), expected=record.word_values.tolist(),
                unavailable=unavailable.tolist(), defined=defined.tolist(),
                end_slots=record.end_slots.tolist(),
                programs=[None if p is None else dict(actions=p.actions.tolist(),
                    frames=p.operation_values.tolist(), leaves=p.leaves.tolist(),
                    names=[(language._compose_binary_rules if k == 1 else
                        language._compose_unary_rules)[op].method_name if k in (1,2) else str(k)
                        for k,op,_ in p.actions.tolist()]) for p in programs])
            with Path(os.environ['INVERSE_DIAGNOSTIC']).open('a') as out:
                out.write(json.dumps(data)+'\n')
        return result
    reader.reconstruct = capture
