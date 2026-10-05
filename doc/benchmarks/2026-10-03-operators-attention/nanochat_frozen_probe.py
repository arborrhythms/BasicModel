"""Held-out native admission cannot consume the evaluator's permanent inventory."""
import copy
import torch
import eval_nanochat_grammar as gate


def test_scoring_does_not_admit_heldout_definitions():
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    before = tuple(model._concept_owner().definitions.word_ids)
    manifest = gate.load_manifest(gate.DEFAULT_MANIFEST)
    result = gate.score_manifest(model, data, manifest, limit=1)
    assert result['metrics']['items'] == 1
    assert tuple(model._concept_owner().definitions.word_ids) == before


def test_checkpoint_clone_preserves_native_ownership_and_scores():
    model, data = gate.build_eval_model(gate.DEFAULT_MODEL, autoload=False)
    trial = copy.deepcopy(model, {id(data): data})
    assert trial._concept_owner()._model is trial
    assert trial._concept_owner().definitions._store() is trial.symbolSpace.ltm
    assert trial.inputSpace.data is data
    item = gate.load_manifest(gate.DEFAULT_MANIFEST)['items'][0]
    with gate.frozen_online_learning(trial):
        a = gate._score_control_batch(trial, data, [item], 'intact', 16)
        b = gate._score_control_batch(trial, data, [item], 'shuffled', 16)
    assert torch.isfinite(a[0]).all() and torch.isfinite(b[0]).all()
    assert not tuple(model._concept_owner().definitions.word_ids)
