"""Native counterexamples to making the existing fold injective by width alone."""
import json
from pathlib import Path
import sys
import tempfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
from identity_experiment import make_model
from Language import IntersectionLayer

DEST = Path(__file__).resolve().parent


def run():
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as patch:
        model = make_model(Path(directory), patch)
        model.eval()
        try:
            with torch.no_grad():
                model._start_spaces_for_forward()
                model._lex_embed_stem(model.inputSpace.prepInput(['itema itemb']))
                owner = model._concept_owner()
                bank = owner.similarity_codebook
                ids = [owner.interpret.word_for_form(word) for word in ('itema', 'itemb')]
                points = [bank.lookup_rows(owner._csw_row_of(identity)).detach().clone()
                          for identity in ids]
                assert all(bool(((point >= 0) & (point <= 1)).all()) for point in points)
                assert not torch.equal(*points)
                questions = []
                op = IntersectionLayer()
                for copies in (1, 2, 8, 32):
                    a, b = (point.repeat(copies).reshape(1, 1, -1) for point in points)
                    ab, ba, aa = op.compose(a, b), op.compose(b, a), op.compose(a, a)
                    questions.append(dict(
                        width=a.shape[-1], native_width=points[0].numel(),
                        widening='repeated coordinates for a diagnostic, not an STM1 implementation',
                        ordered_inputs_distinct=True, swap_outputs_equal=bool(torch.equal(ab, ba)),
                        duplicate_inputs_distinct=True, duplicate_outputs_equal=bool(torch.equal(a, aa)),
                        swap_max_error=float((ab - ba).abs().max()),
                        duplicate_max_error=float((a - aa).abs().max())))
                report = dict(
                    training_steps=0, seed_selection='none',
                    bank_capacity=len(bank.W), native_identities=ids,
                    result='existing_fold_not_injective_at_any_width',
                    explanation='Coordinatewise intersection commutes and is idempotent. '
                                'Bank cleanup cannot choose uniquely between inputs with identical outputs.',
                    questions=questions)
                (DEST / 'fold-audit.json').write_text(json.dumps(report, indent=2) + '\n')
                text = '# STM1 pre-implementation fold audit\n\n'
                text += ('No training, artifact loading, seed selection or gate relaxation. '
                         'Native codes are read from the current model. Wider repeated-coordinate '
                         'examples diagnose the existing kernel; they are not a proposed STM1 code.\n\n'
                         'For every width, intersection(a, b) = intersection(b, a), and '
                         'intersection(a, a) = a. Exact ordered leaf-occurrence recovery therefore '
                         'needs an order/multiplicity encoding as well as a width. The finite-bank '
                         'clean-up index cannot resolve identical forward outputs. This does not '
                         'reject invertible STM1; it identifies the missing structural law.\n')
                for i, question in enumerate(questions, 1):
                    text += f'\n## Question {i}: is the existing fold injective at width {question["width"]}?\n\n'
                    text += '| Measurement | Result |\n|---|---|\n'
                    for key, value in question.items():
                        text += f'| {key} | {value} |\n'
                (DEST / 'fold-audit.md').write_text(text)
                print(json.dumps({key: value for key, value in report.items()
                                  if key != 'questions'}, indent=2))
        finally:
            model.End()
            model.symbolSpace.soft_reset()


if __name__ == '__main__':
    run()
