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
        # What was expected at the look of the hidden word, before that
        # look's result is read: each look writes it anew (ConceptSpread
        # .write_look) after its forecast of the look's extent.
        import ConceptSpread
        from LookForecast import LookForecast
        from reading_fixtures import SourceOrder
        # The words are read in their order (a supplied reading).
        monkeypatch.setattr(model, '_attention_supplied_reading', SourceOrder(), raising=False)
        looks, extents = [], []
        forecast, write = LookForecast.forecast, ConceptSpread.write_look
        def forecast_extent(self, extent, looked=None):
            _, inputs = self._outside(extent, looked)
            came = inputs[0].amax(-1) > 0 if len(inputs) else None
            extents.append((None, extent.detach().clone(), came))
            return forecast(self, extent, looked)
        def write_kept(owner_, rows, value):
            write(owner_, rows, value)
            looks.append((extents[-1], owner_._expected_knowing[0].detach().clone()))
        monkeypatch.setattr(LookForecast, 'forecast', forecast_extent)
        monkeypatch.setattr(ConceptSpread, 'write_look', write_kept)
        from CandidateGrowth import matrices, pair_rows
        table = owner._concept_allocator.symbol_table()
        print(dict(candidates={name: held.counts() for name, held in matrices(owner).items()},
                   pair_rows=list(pair_rows(owner)), symbols=list(zip(table.handles, map(table.members, range(len(table)))))))
        for text, hidden_row in zip(examples, hidden_rows):
            looks.clear(); extents.clear()
            read(model, text)
            assert owner._observed_knowing[0, hidden_row].sum() == 0
            # The look whose extent holds the hidden word (the second word)
            # and not the word that came.
            # The look of the hidden word: the look that asked for it, whose
            # extent holds no reading of the word that came (the hidden word
            # lights nothing of its own).
            at = [(knowing, came) for (_, extent, came), knowing in looks
                  if not bool(extent[0, 0]) and came is not None]
            assert at, 'no look of the hidden word'
            knowing, came = at[0]
            values = knowing[:owner._order_caps()[0], 0]
            # Absent: neither the hidden word nor what the earlier looks read.
            absent = torch.ones_like(values, dtype=torch.bool)
            absent[:len(came)] &= ~came
            absent[hidden_row] = False
            print(dict(presentations=200, text=text, hidden_row=hidden_row,
                       looks=[extent[0].tolist() for (_, extent, _), _ in looks], came=came.nonzero().flatten().tolist(),
                       expected=float(values[hidden_row]), other_absent=float(values[absent].max())))
            assert values[hidden_row] > values[absent].max()
            assert not owner._expected_knowing.requires_grad
    finally:
        model.omission.draw = ordinary
        from assessment_support import record
        record(model, dict(word='alpha', words=['alpha', 'gamma', 'beta', 'delta']))
        model.End(); model.symbolSpace.soft_reset()
