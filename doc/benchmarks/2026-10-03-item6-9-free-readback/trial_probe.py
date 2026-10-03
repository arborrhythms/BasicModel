"""Actual trial comparison and discarded-row reader-credit probes."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]


def test_trial_comparison_is_reconstruction_only_and_reader_rows_are_selected(monkeypatch):
    import Models, util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, data = _fresh_model(str(ROOT/'data/XOR_grammar.xml'))
    costs, masks = [], []
    score = model._sentence_path_cost
    train = model._sentence_train_step
    def observed_score(*args, **kwargs):
        result = score(*args, **kwargs)
        costs.append(model._sentence_cost_registry.total(objective='reconstruction').detach().clone())
        return result
    def observed_train(loss):
        mask = getattr(model, '_sentence_reader_rows', None)
        masks.append(None if mask is None else mask.detach().clone())
        return train(loss)
    monkeypatch.setattr(model, '_sentence_path_cost', observed_score)
    monkeypatch.setattr(model, '_sentence_train_step', observed_train)
    try:
        optimizer=model.getOptimizer(lr=.01)
        raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
        model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',
            batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
        torch.testing.assert_close(model._sentence_trial_costs[0], torch.stack(costs,-1),rtol=0,atol=0)
        assert len(masks)==2 and all(mask is not None for mask in masks)
        wins=model._sentence_winners[0]
        torch.testing.assert_close(masks[0],~wins)
        torch.testing.assert_close(masks[1],wins)
    finally:
        model.End()
