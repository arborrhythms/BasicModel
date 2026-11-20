"""Expectation restores an omitted word from the word that came (sigma-pi §21, certificate 57 (b))."""
from pathlib import Path
import pytest
import torch

pytestmark = pytest.mark.assessment_rung(1)


def lesson(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32,
        concept_rows=32, training_overrides={'reconstructionPlacement': 'eager'})
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def read(model, text, *, training=False, optimizer=None):
    source = model.inputSpace.prepInput([text])
    return model.runBatch(train=training, optimizer=optimizer or model.getOptimizer(lr=.003), batchSize=1,
                         batch_override=(source, torch.empty(1, 0)))


def omit_second(monkeypatch, model):
    from Omission import Draw
    def draw(valid, *, training):
        hidden = torch.zeros_like(valid)
        hidden[:, 1] = valid[:, 1]
        return Draw(hidden.any(-1), hidden, torch.zeros(len(valid), dtype=torch.long))
    monkeypatch.setattr(model.omission, 'draw', draw)


def test_81_expectation_restores_the_omitted_native_word(tmp_path, monkeypatch):
    from ConceptSpread import rule_order
    model = lesson(tmp_path, monkeypatch)
    ordinary = model.omission.draw
    try:
        from PrependedSymbols import word_view
        examples = ('alpha gamma', 'beta delta')
        hidden_rows = []
        # MTM nominates a form on its concept_mint_count-th reading (sigma-pi
        # §21, rule 20): each example is read until its words exist.
        from nomination_fixtures import readings_before_nomination
        for text in examples:
            readings_before_nomination(model, lambda: read(model, text))
            read(model, text)
            hidden_rows.append(int(word_view(model.inputSpace, 'concept_rows')[0, 1]))
        assert all(row >= 0 for row in hidden_rows)
        owner = model._concept_owner()
        omit_second(monkeypatch, model)
        optimizer = model.getOptimizer(lr=.01)
        for presentation in range(200):
            read(model, examples[presentation % 2], training=True, optimizer=optimizer)
        for text, hidden_row in zip(examples, hidden_rows):
            read(model, text)
            assert owner._observed_knowing[0, hidden_row].sum() == 0
            values = owner._expected_knowing[0, :owner._order_caps()[0], 0]
            absent = owner._observed_knowing[0, :len(values)].amax(-1) == 0
            absent[hidden_row] = False
            assert values[hidden_row] > values[absent].max()
            assert hidden_row in rule_order(owner, 0)
            assert not owner._expected_knowing.requires_grad
            print(dict(presentations=200, text=text, hidden_row=hidden_row,
                       expected=float(values[hidden_row]), other_absent=float(values[absent].max())))
    finally:
        model.omission.draw = ordinary
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma', 'beta', 'delta']))
        model.End(); model.symbolSpace.soft_reset()
