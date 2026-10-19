"""Explicit trained-artifact prerequisites for learning-quality tests."""
import pytest
from LearningEvaluation import MIN_FINEWEB_SENTENCES, checkpoint_readiness


@pytest.fixture
def fineweb_checkpoint(phase_artifact, phase_registry):
    if not phase_artifact.config:
        pytest.fail('FineWeb phase artifact requires its model config')
    config = phase_registry.resolve(phase_artifact.config)
    checkpoint = phase_registry.resolve(phase_artifact.path)
    minimum = phase_artifact.minimum_sentences or MIN_FINEWEB_SENTENCES
    if not config.is_file():
        pytest.fail('the registered FineWeb config does not exist')
    readiness = checkpoint_readiness(checkpoint, minimum_sentences=minimum)
    if not readiness.eligible:
        pytest.fail(f'registered FineWeb artifact ineligible: {readiness.reason}', pytrace=False)
    return dict(config=config, checkpoint=checkpoint, minimum_sentences=minimum)


@pytest.fixture
def fineweb_trained_model(fineweb_checkpoint):
    from eval_fineweb_learning import load_model
    model = load_model(**fineweb_checkpoint)
    try:
        yield model
    finally:
        model.End()
        model.symbolSpace.soft_reset()
