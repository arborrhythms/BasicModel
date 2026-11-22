"""The forecast at a look depends on observations outside its extent."""
from dataclasses import replace
import torch


def test_inside_changes_the_read_but_not_the_forecast(tmp_path, monkeypatch):
    from test_item18_expect import lesson, read
    model = lesson(tmp_path, monkeypatch)
    try:
        # MTM nominates a form on its concept_mint_count-th reading (sigma-pi
        # §21, rule 20): the look's read is of words that exist.
        from nomination_fixtures import readings_before_nomination
        readings_before_nomination(model, lambda: read(model, 'alpha gamma'))
        read(model, 'alpha gamma')
        recorded = model._sentence_look_fields[0][0]
        assert all('surprise' in entry for entry in recorded)
        from LookForecast import LookForecast
        owner = model._concept_owner()
        opened = recorded[0]['field']
        readings = opened.evidence.shape[2]
        extent = torch.zeros(1, readings, dtype=torch.bool)
        extent[0, 0] = True
        changed = replace(opened, evidence=torch.where(extent[None, ..., None],
                                                       1 - opened.evidence, opened.evidence))
        shared = owner._expected_knowing.clone()
        first = LookForecast(owner, opened, opened.where)
        second = LookForecast(owner, changed, opened.where)
        torch.testing.assert_close(first.forecast(extent), second.forecast(extent), rtol=0, atol=0)
        actual = first.read(first._located(opened), extent)
        other = second.read(second._located(changed), extent)
        assert not torch.equal(actual, other)
        torch.testing.assert_close(owner._expected_knowing, shared, rtol=0, atol=0)
        assert first.latest.shape == (1, 2)
        # A look's surprise replaces the latest; a stream with no look keeps it.
        replacement = first.observe(opened, extent, torch.ones_like(first.forecast(extent)))
        torch.testing.assert_close(first.latest, replacement)
        before = first.latest.clone()
        first.observe(opened, torch.zeros_like(extent), first.forecast(extent))
        torch.testing.assert_close(first.latest, before)
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_excess_and_omission_keep_independent_lanes_and_magnitudes():
    from LookForecast import LookForecast
    look = torch.tensor([[True]])
    read = torch.tensor([[[.5, 0.], [0., .375]]])
    torch.testing.assert_close(LookForecast.surprise(read, read, look), torch.zeros(1, 2))
    silent = torch.zeros_like(read)
    torch.testing.assert_close(LookForecast.surprise(read, silent, look), torch.tensor([[.875, 0.]]))
    torch.testing.assert_close(LookForecast.surprise(silent, read, look), torch.tensor([[0., .875]]))
    forecast = torch.tensor([[[.25, .5], [0., .125]]])
    torch.testing.assert_close(LookForecast.surprise(read, forecast, look), torch.tensor([[.5, .5]]))


def test_glance_predicts_each_stream_in_one_batch(tmp_path, monkeypatch):
    import ConceptSpread
    from LookForecast import LookForecast
    from test_item18_expect import lesson, read

    model = lesson(tmp_path, monkeypatch)
    try:
        read(model, 'alpha gamma')
        owner = model._concept_owner()
        field = model._sentence_open_fields[0]
        spans = torch.cat((field.where, torch.zeros_like(field.where[:, :1])), 1)
        batches = []
        original = ConceptSpread.expected

        def observe_call(owner, evidence):
            batches.append(evidence.shape[0])
            return original(owner, evidence)

        monkeypatch.setattr(ConceptSpread, 'expected', observe_call)
        forecast = LookForecast(owner, field, spans)
        assert batches == [1]
        assert forecast.latest.shape == (1, 2)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
