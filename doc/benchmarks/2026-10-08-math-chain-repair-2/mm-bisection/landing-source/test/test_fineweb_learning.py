"""Quality is evaluated after substantial training, not on random tiny models."""
import statistics
import pytest
from eval_fineweb_learning import load_model, read_validation


@pytest.mark.slow
@pytest.mark.artifact_eval
def test_trained_fineweb_prediction_benefits_from_context(fineweb_trained_model):
    report = read_validation(fineweb_trained_model)
    # A checkpoint ablation, not a claim about separately trained controls.
    # Require some benefit, with no arbitrary percentage of improvement.
    for controls in report['controls'].values():
        objective = lambda c: c['feature_mse'] + c['presence_bce']
        assert objective(controls['ordered']) < objective(controls['shuffled']), report
        assert objective(controls['ordered']) < objective(controls['context_free']), report


@pytest.mark.slow
@pytest.mark.artifact_eval
def test_trained_fineweb_remainder_reduces_work_at_matched_error(fineweb_checkpoint):
    trials = {}
    for gain in (0., 1.):
        model = load_model(**fineweb_checkpoint)
        try:
            trials[gain] = read_validation(model, gain=gain)['thought']
        finally:
            model.End()
            model.symbolSpace.soft_reset()
    full, remainder = trials[0.], trials[1.]
    assert full and len(full) == len(remainder), trials
    assert [(r['row'], r['document']) for r in full] == [
        (r['row'], r['document']) for r in remainder], trials
    assert all(r['answered'] for r in full + remainder), trials
    mean = lambda rows, key: statistics.mean(r[key] for r in rows)
    for key in ('feature_mse', 'presence_bce'):
        assert mean(remainder, key) <= mean(full, key) + 1e-6, trials
    assert mean(remainder, 'work') < mean(full, 'work'), trials
