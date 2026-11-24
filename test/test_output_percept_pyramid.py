"""Read the normal writer's result, then change only what the scorer expects."""
import torch

from test_item17_echo import echo, echo_model
from test_item17_places import nominate


def test_emitted_percepts_charge_content_place_and_silence(tmp_path, monkeypatch, eager_reading):
    model = echo_model(tmp_path, monkeypatch)
    score = model._output_target_cost
    comparisons = []
    gradients = []
    walk_gradients = []
    sigma_gradients = []
    requests = []
    state = model.conceptualSpace.stm.model_state
    observe = state.observe
    units = model.request_read.units

    def observe_extent(spans, index):
        observe(spans, index)
        for row in (index >= 0).nonzero().flatten():
            torch.testing.assert_close(state.last_where[row], spans[row, index[row]])

    def read_state(x, previous):
        location = state.last_where.clone()
        endpoints = location.clone()
        endpoints[:, 1] -= 1
        expected = model.where_registry.band(endpoints).flatten(1)
        torch.testing.assert_close(previous[:, :-4], expected)
        torch.testing.assert_close(previous[:, -4:], model.present_code().expand(len(x), -1))
        requests.append(location)
        return units(x, previous)

    def scored(output, targets):
        result = score(output, targets)
        if output.texts == ('Felix.',):
            costs = [score(output, [wanted])[0] for wanted in ('Felix.', 'Fylix.', '.Felix', ' Felix.', 'Felix', '')]
            comparisons.append([float(cost.detach()) for cost in costs])
            if output.byte_amounts.requires_grad:
                (gradient,) = torch.autograd.grad(costs[1], output.byte_amounts, retain_graph=True)
                gradients.append(float(gradient.abs().sum()))
                # A selected reference can own a fixed spelling. Measure which
                # continuous paths the returned answer actually trains.
                sigma = model._concept_owner().install_order_zero()
                (update,) = torch.autograd.grad(costs[1], sigma.value, retain_graph=True, allow_unused=True)
                if update is not None:
                    sigma_gradients.append(float(update.abs().sum()))
                if output.concepts.requires_grad:
                    (walk,) = torch.autograd.grad(costs[1], output.concepts, retain_graph=True, allow_unused=True)
                    if walk is not None:
                        walk_gradients.append(float(walk.abs().sum()))
        return result

    try:
        nominate(model, ('Felix', '.', 'Felix.'))
        monkeypatch.setattr(model.inputSpace.data, 'has_supervised_outputs', True)
        monkeypatch.setattr(model.inputSpace.data, 'text_answers', {'train': ['Felix.'], 'validation': []})
        monkeypatch.setattr(model, '_output_target_cost', scored)
        monkeypatch.setattr(state, 'observe', observe_extent)
        monkeypatch.setattr(model.request_read, 'units', read_state)
        batch = echo(model, 0, 'Felix.', train=True, optimizer=model.getOptimizer())
        assert bytes(batch.outputPred.detach().flatten().tolist()).rstrip(b'\0') == b'Felix.'
        assert comparisons and all(correct < min(errors) for correct, *errors in comparisons)
        assert gradients and max(gradients) > 0
        assert sigma_gradients and max(sigma_gradients) > 0
        assert any(bool((extent[:, 0] >= 0).any()) for extent in requests)
        print('Felix. exact, substituted, reordered, shifted, missing mark, silence:', comparisons[-1])
        print('Gradient L1 at emitted bytes, order-zero sigma and walk leaves:',
              max(gradients), max(sigma_gradients), max(walk_gradients, default=0.))
    finally:
        model.End()


def test_returned_words_pay_for_a_transposition_with_the_same_byte_counts(tmp_path, monkeypatch, eager_reading):
    from test_item17_echo import answer_text
    texts = ('This That', 'That This')
    model = echo_model(tmp_path, monkeypatch)
    try:
        nominate(model, ('This', 'That', *texts))
        data = model.inputSpace.data
        monkeypatch.setattr(data, 'has_supervised_outputs', True)
        monkeypatch.setattr(data, 'text_answers', {'train': texts})
        for index, text in enumerate(texts):
            batch = echo(model, index, text, train=False, optimizer=None)
            assert answer_text(batch.outputPred) == text
            answer = model._last_answer_construction
            with torch.no_grad():
                exact = model._output_target_cost(answer, [text])
                transposed = model._output_target_cost(answer, [texts[1 - index]])
            assert exact.item() < transposed.item()
            model.End()
    finally:
        model.End()
