"""A restored sentence trial has the same grad contract as later words."""
import torch


def test_first_and_later_words_keep_the_same_grad_carry_contract(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8')
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    # This checks the word brick's input contract. The existing compiled
    # forward/interleave cases cover compilation and tied reconstruction.
    model.reconstruct_in_loop = False
    model.loss.reconstruction_scale = 0.
    original = model._run_sentence_word_bricks
    seen = []

    def driver(*args):
        compose = args[-1]

        def checked(payload, index, unused, lang, stm):
            carries = tuple(v for bank in (lang, stm) for v in bank
                            if v.is_floating_point())
            assert all(v.requires_grad for v in carries), (
                'restored and later word carries must all require grad')
            seen.append((model._sentence_trial, int(index)))
            return compose(payload, index, unused, lang, stm)

        return original(*args[:-1], checked)

    monkeypatch.setattr(model, '_run_sentence_word_bricks', driver)
    try:
        inputs = model.inputSpace.prepInput(['alpha beta', 'gamma delta'])
        model.runBatch(train=True, batchSize=2, optimizer=model.getOptimizer(lr=.003),
            batch_override=(inputs, torch.empty(2, 0)))
        assert set(seen) == {('exploit', 0), ('exploit', 1),
                             ('explore', 0), ('explore', 1)}
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
