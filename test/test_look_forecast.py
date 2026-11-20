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
        span = opened.where[:, :1]
        valid = (opened.where == span).all(-1)
        changed = replace(opened, evidence=torch.where(valid[None,...,None],
                                                       1 - opened.evidence, opened.evidence))
        shared = owner._expected_knowing.clone()
        first = LookForecast(owner, opened, opened.where)
        second = LookForecast(owner, changed, opened.where)
        torch.testing.assert_close(first.forecast(span), second.forecast(span), rtol=0, atol=0)
        actual = first.read(first._located(opened), opened.where, span)
        other = second.read(second._located(changed), changed.where, span)
        assert not torch.equal(actual, other)
        torch.testing.assert_close(owner._expected_knowing, shared, rtol=0, atol=0)
        assert first.latest.shape == (*opened.where.shape[:2], 2)
        # The other extent must retain its original pair after this look.
        before = first.latest.clone()
        observed = replace(opened, where=span, evidence=opened.evidence[:, :, :1], when=opened.when[:, :1])
        replacement = first.observe(observed, span, torch.ones_like(first.forecast(span)))
        assert not torch.equal(before[:, :1], replacement)
        torch.testing.assert_close(first.latest[:, :1], replacement)
        torch.testing.assert_close(first.latest[:, 1:], before[:, 1:])
        whole = torch.stack((opened.where[..., 0].amin(1), opened.where[..., 1].amax(1)), -1)[:, None]
        torch.testing.assert_close(first.for_brackets(whole)[:, 0], first.latest.sum(1))
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_excess_and_omission_keep_independent_lanes_and_magnitudes():
    from LookForecast import difference
    read = torch.tensor([[[[.5, 0.], [0., .375]]]])
    torch.testing.assert_close(difference(read, read), torch.zeros(1, 1, 2))
    silent = torch.zeros_like(read)
    torch.testing.assert_close(difference(read, silent), torch.tensor([[[.875, 0.]]]))
    torch.testing.assert_close(difference(silent, read), torch.tensor([[[0., .875]]]))
    forecast = torch.tensor([[[[.25, .5], [0., .125]]]])
    torch.testing.assert_close(difference(read, forecast), torch.tensor([[[.5, .5]]]))


def test_zero_started_projection_preserves_bracket_logits_and_can_learn():
    from AttentionContext import project
    from Language import OperationSelectionLayer
    chooser = OperationSelectionLayer(d_model=2)
    keys = torch.tensor([[[1., 0.], [0., 1.]]])
    allowed = torch.ones(1, 2, 6, dtype=torch.bool)
    spaces = torch.zeros(1, 2, dtype=torch.long)
    prefix = torch.ones(1, 3, requires_grad=True)
    surprise = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    mapping = torch.nn.Linear(5, 2, bias=False)
    torch.nn.init.zeros_(mapping.weight)
    def scores(context):
        return chooser.attend(keys, allowed, spaces, context=context, return_details=True)[3]['logits']
    actual = scores(project(mapping, prefix, surprise))
    torch.testing.assert_close(actual, scores(None), rtol=0, atol=0)
    # Once an act has a learned key, its gradient reaches the surprise columns.
    with torch.no_grad():
        chooser.bracket_anchor[0, 0] = 1.
    scores(project(mapping, prefix, surprise))[0, 0].backward()
    assert mapping.weight.grad[:, -2:].count_nonzero()
    assert prefix.grad is None and surprise.grad is None


def test_glance_predicts_each_live_extent_in_one_batch(tmp_path, monkeypatch):
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
        live = spans[..., 1] > spans[..., 0]
        assert int(live.sum()) >= 2
        assert batches == [int(live.sum())]
        assert not forecast.latest[~live].any()
        whole = torch.stack((spans[:, 0, 0], spans[..., 1].amax(-1)), -1)[:, None]
        initial = forecast.latest.clone()
        inside = spans[:, :1].clone()
        inside[..., 1] -= 1
        assert (inside[..., 1] > inside[..., 0]).all()
        for selected in (whole, inside):
            prediction = torch.ones_like(forecast.forecast(selected))
            pair = forecast.observe(field, selected, prediction)
            assert pair.shape == (1, 1, 2) and pair.any()
            torch.testing.assert_close(forecast.latest, initial, rtol=0, atol=0)
            torch.testing.assert_close(forecast.for_brackets(whole)[:, 0], initial.sum(1))
    finally:
        model.End()
        model.symbolSpace.soft_reset()
