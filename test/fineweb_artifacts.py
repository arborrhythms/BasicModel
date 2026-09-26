"""Explicit trained-artifact prerequisites for learning-quality tests."""
import os
from pathlib import Path
import pytest
from LearningEvaluation import MIN_FINEWEB_SENTENCES, checkpoint_readiness


@pytest.fixture
def fineweb_checkpoint():
    checkpoint = os.environ.get('BASICMODEL_FINEWEB_CHECKPOINT')
    if not checkpoint:
        pytest.skip('requires BASICMODEL_FINEWEB_CHECKPOINT with at least '
                    f'{MIN_FINEWEB_SENTENCES:,} completed FineWeb training sentences')
    config = os.environ.get('BASICMODEL_FINEWEB_CONFIG')
    if not config:
        pytest.fail('BASICMODEL_FINEWEB_CHECKPOINT requires its BASICMODEL_FINEWEB_CONFIG')
    minimum = int(os.environ.get('BASICMODEL_FINEWEB_MIN_SENTENCES', MIN_FINEWEB_SENTENCES))
    config, checkpoint = Path(config).resolve(), Path(checkpoint).resolve()
    if not config.is_file() or not checkpoint.is_file():
        pytest.fail('the explicitly configured FineWeb checkpoint/config does not exist')
    readiness = checkpoint_readiness(checkpoint, minimum_sentences=minimum)
    if not readiness.eligible:
        pytest.skip(readiness.reason)
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
