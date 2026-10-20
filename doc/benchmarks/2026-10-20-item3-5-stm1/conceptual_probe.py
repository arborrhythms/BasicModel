"""Production conceptual meet after one unchanged four-sentence bootstrap.

No optimizer, replacement training, seed selection or retained phase artifact.
The earlier G0 measured the paired carrier; this separates its two metrics.
"""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]

from IdentityRows import coordinates, meet_ratios
from test_mm_xor import _fresh_model


def run():
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        with torch.no_grad():
            # The standing bootstrap fixture's one presentation, unchanged.
            model.forward(model.inputSpace.prepInput([
                'hello world', 'hello there', 'loving world', 'loving there']))
            owner = model._concept_owner()
            bank = owner.similarity_codebook
            derived = bank.mereology
            derived.begin_forward()
            rows = sorted(derived._context)
            raw = bank.lookup_rows(torch.tensor(rows, device=bank.W.device))
            semantic = coordinates(owner, raw)
            assert len(rows) == 4 and len(torch.unique(semantic, dim=0)) == 4
            assert bool(((semantic >= 0) & (semantic <= 1)).all())
            assert not semantic[:, :derived.percept_event_width].any()
            q = semantic[0]
            candidates = [torch.maximum(q, semantic[i]) for i in (1, 2)]
            records = []
            for number, c in enumerate(candidates, 1):
                candidate, request = meet_ratios(q, c)
                records.append(dict(question=f'request / enriched candidate {number}',
                    candidate_coverage=float(candidate), request_coverage=float(request),
                    distance=float((q-c).norm()), request_mass=float(q.sum()),
                    candidate_mass=float(c.sum()), meet_equals_request=bool(
                        torch.equal(torch.minimum(q, c), q))))
                assert request == 1
            changed = raw.clone()
            changed[:, :derived.percept_width] = 1 - changed[:, :derived.percept_width]
            torch.testing.assert_close(coordinates(owner, changed), semantic, rtol=0, atol=0)
            result = dict(training_steps=0, presentations=1, seed_selection='none',
                carrier_width=raw.shape[-1], form_width=derived.percept_width,
                band_end=derived.percept_event_width, meaning_width=derived.context_width,
                distinct_meanings=4, minimum=float(semantic.min()), maximum=float(semantic.max()),
                form_change_max_semantic_difference=float(
                    (coordinates(owner, changed)-semantic).abs().max()),
                rows=rows, conceptual_points=semantic.tolist(), questions=records)
            (DEST / 'conceptual-probe.json').write_text(json.dumps(result, indent=2)+'\n')
            text = '# G0 follow-up: the conceptual metric\n\n'
            text += ('One presentation of the standing four-sentence bootstrap, no optimizer. '
                     'The original G0 paired-carrier measurement remains historical; it is not '
                     'promoted to evidence about the isolated conceptual metric.\n\n')
            text += (f'Carrier width {result["carrier_width"]}; form width {result["form_width"]}; '
                     f'conceptual complement {result["meaning_width"]}. The four meanings are '
                     'distinct. Changing every form coordinate changes the conceptual read by '
                     f'{result["form_change_max_semantic_difference"]}. These are constructed '
                     'containment comparisons over native meanings, not learned reference accuracy.\n')
            for record in records:
                text += f'\n## {record["question"]}\n\n| Measurement | Value |\n|---|---:|\n'
                for key, value in record.items():
                    if key != 'question':
                        text += f'| {key} | {value} |\n'
            (DEST / 'conceptual-probe.md').write_text(text)
            print(json.dumps({k:v for k,v in result.items() if k!='conceptual_points'}, indent=2))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


if __name__ == '__main__':
    run()
