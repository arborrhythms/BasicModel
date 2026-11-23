"""``attend``'s read of the conceptual field (sigma-pi §21, rule 80; round
6d, eleventh pass): one network every use shares, its request a mask in 0
to 1 at the concepts' rows, the stopping decision; the
field placed at the concepts' rows and the last look's field spread once."""
import pytest
import torch


def test_the_request_starts_small_on_every_concept_and_is_taught_from_its_read():
    from RequestRead import RequestRead
    read = RequestRead(6, 8)
    with torch.no_grad():
        read.ask.bias.fill_(-4.)
    x = read.given(torch.rand(2, 6, 2), torch.rand(2, 6, 2), torch.ones(2))
    assert x.shape == (2, 4 * 6 + 1)
    units = read.units(x)
    request = read.request(units)
    assert request.shape == (6, 2, 2) and not request[..., 1].any()
    torch.testing.assert_close(request[..., 0], torch.full((6, 2), torch.sigmoid(torch.tensor(-4.)).item()))
    assert not read.stopping(units).any()
    # The request is a logistic of a linear read: a cost on it reaches the
    # read's top and, once the top is not nothing, the units below it.
    request[3, 0, 0].backward()
    assert read.ask.bias.grad[3] > 0 and read.ask.weight.grad[3].abs().sum() > 0
    assert read.first.weight.grad is None or not read.first.weight.grad.any()


def test_a_field_is_placed_at_its_concepts_rows_and_spread_once():
    from types import SimpleNamespace
    from RequestRead import at_rows
    evidence = torch.zeros(3, 1, 2, 2)
    evidence[0, 0, 0] = torch.tensor([.4, 0.])
    evidence[0, 0, 1] = torch.tensor([.7, .1])
    evidence[2, 0, 1] = torch.tensor([1.5, .2])
    field = SimpleNamespace(evidence=evidence)
    held = at_rows(torch.tensor([4, -1, 1]), field, 5, 1)
    assert held.shape == (1, 5, 2)
    torch.testing.assert_close(held[0, 4], torch.tensor([.7, .1]))
    torch.testing.assert_close(held[0, 1], torch.tensor([1., .2]))
    assert not held[0, [0, 2, 3]].any()


def test_sentence_training_reaches_the_request_in_the_model(tmp_path, monkeypatch):
    """Measure whether ordinary sentence training reaches the attention request."""
    import SentenceLooks
    from test_item17_places import minimal, model_for, nominate
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        asked, scored = [], []
        original, activity = SentenceLooks.attend, SentenceLooks.word_activity
        def requested(model, units):
            request = original(model, units)
            request.retain_grad()
            asked.append(request)
            return request
        def word_activity(*args):
            words = activity(*args)
            if words.requires_grad:
                words.retain_grad()
            scored.append(words)
            return words
        monkeypatch.setattr(SentenceLooks, 'attend', requested)
        monkeypatch.setattr(SentenceLooks, 'word_activity', word_activity)
        optimizer = model.getOptimizer(lr=.01)
        reached = []
        for presentation in range(3):
            asked.clear(); scored.clear()
            read(model, 'This is Felix.', training=True, optimizer=optimizer)
            model.End()
            grad = lambda t: 0. if t.grad is None else float(t.grad.abs().sum())
            print(dict(presentation=presentation, request=[round(grad(t), 6) for t in asked],
                       words=[round(grad(t), 6) for t in scored]))
            if presentation:
                reached.append(max(grad(t) for t in asked[:2]))
        assert max(reached) > 0.
    finally:
        model.End()


@pytest.mark.parametrize('grammar', (True, False), ids=('sentence', 'grammar-free'))
def test_two_words_train_the_sampled_request_from_the_following_observation(tmp_path, monkeypatch, eager_reading, grammar):
    from LookForecast import OccurrenceCredit
    from test_item17_places import model_for, nominate
    from test_item18_expect import read
    from test_item17_one_reconstruction import _grammar_free_text_model
    build = model_for if grammar else _grammar_free_text_model
    model = build(tmp_path, monkeypatch)
    try:
        nominate(model, ('This', 'is', 'This is'))
        recorded = []
        register = OccurrenceCredit.register

        def capture(credit, registry):
            register(credit, registry)
            if 'attention.next_observation' in registry._terms:
                recorded.append(len(credit.terms))

        monkeypatch.setattr(OccurrenceCredit, 'register', capture)
        before = model.request_read.ask.weight.detach().clone()
        optimizer = model.getOptimizer(lr=.01)
        read(model, 'This is', training=True, optimizer=optimizer)
        assert recorded == [1]
        assert not torch.equal(model.request_read.ask.weight.detach(), before)
    finally:
        model.End()
