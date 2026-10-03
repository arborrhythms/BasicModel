"""Raw forward and evaluation share the training sentence boundary."""
import copy
import torch
import pytest


@pytest.mark.parametrize('entry', ['forward_train_mode', 'forward_eval_mode', 'batch_eval'])
def test_every_entry_writes_once_with_one_trial_and_no_optimizer(tmp_path, monkeypatch, entry):
    from test_meronomy_ladder import _build_ladder_variant
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'single_boundary', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>'),
        ('<serialWordCapacity>8</serialWordCapacity>', '<serialWordCapacity>16</serialWordCapacity>'),
        ('<serialWordBuckets>8</serialWordBuckets>', '<serialWordBuckets>16</serialWordBuckets>')])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.reconstruct_in_loop = False
    model.loss.reconstruction_scale = 0.
    def forbidden(*args, **kwargs):
        raise AssertionError('inference invoked the optimizer')
    monkeypatch.setattr(model, '_sentence_train_step', forbidden)
    writes = []
    store = model.symbolSpace.ltm_store
    original = store.write_clause
    def write(*args, **kwargs):
        writes.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'write_clause', write)
    try:
        inputs = model.inputSpace.prepPackedInput([['1 plus 2', '3 plus 4']])
        model.train(entry == 'forward_train_mode')
        before = {name: value.detach().clone() for name, value in model.languageSpace._tree_layer(2).chooser.named_parameters()}
        if entry == 'batch_eval':
            model.runBatch(train=False, batchSize=1, split='runtime',
                           batch_override=(inputs, torch.zeros(1, 1, 0)))
        else:
            model(inputs)
        assert len(writes) == 2
        assert set(model._sentence_fields) == {0, 1}
        assert all(cost.shape == (1, 1) for cost in model._sentence_trial_costs)
        assert model._open_sentence_slot is None
        assert not model._reconstruction_stack()._choice_mask.any()
        for name, value in model.languageSpace._tree_layer(2).chooser.named_parameters():
            if name in before:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
