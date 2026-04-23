"""Unit tests for SentencePrimingLayer."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

from Layers import SentencePrimingLayer


class _FakeDiscourse:
    def __init__(self, predicted=None, confidence=None, bias=None):
        self._predicted = predicted
        self._confidence = confidence
        self._bias = bias
        self.predict_calls = 0

    def predict(self):
        self.predict_calls += 1
        return self._predicted, self._confidence

    def prime(self, predicted, confidence, scale):
        if predicted is None or confidence is None:
            return None
        return self._bias * float(scale)


class _FakeWordSpace:
    def __init__(self, discourse):
        self.discourse = discourse


def test_primer_adds_bias_when_discourse_present():
    ws = _FakeWordSpace(_FakeDiscourse(
        predicted=torch.zeros(4), confidence=torch.tensor(0.5),
        bias=torch.ones(4),
    ))
    layer = SentencePrimingLayer(wordSpace_ref=lambda: ws, scale=0.1)
    x = torch.zeros(2, 3, 4)
    y = layer(x)
    expected = torch.full_like(x, 0.1)
    assert torch.allclose(y, expected), (
        f"expected uniform 0.1 bias, got {y.unique()}")


def test_primer_passthrough_when_no_discourse():
    ws = _FakeWordSpace(None)
    layer = SentencePrimingLayer(wordSpace_ref=lambda: ws, scale=0.1)
    x = torch.randn(2, 3, 4)
    y = layer(x)
    assert torch.equal(x, y)


def test_primer_fires_once_per_sentence():
    disc = _FakeDiscourse(
        predicted=torch.zeros(4), confidence=torch.tensor(1.0),
        bias=torch.ones(4),
    )
    ws = _FakeWordSpace(disc)
    layer = SentencePrimingLayer(wordSpace_ref=lambda: ws, scale=1.0)
    x = torch.zeros(1, 1, 4)
    _ = layer(x)
    _ = layer(x)  # second call in same sentence
    assert disc.predict_calls == 1, (
        "discourse.predict() should be called once per sentence, not per position")


def test_primer_reset_rearms():
    disc = _FakeDiscourse(
        predicted=torch.zeros(4), confidence=torch.tensor(1.0),
        bias=torch.ones(4),
    )
    ws = _FakeWordSpace(disc)
    layer = SentencePrimingLayer(wordSpace_ref=lambda: ws, scale=1.0)
    _ = layer(torch.zeros(1, 1, 4))
    layer.Reset()
    _ = layer(torch.zeros(1, 1, 4))
    assert disc.predict_calls == 2, "Reset() should re-arm the predictor for the next sentence"


def test_primer_skips_when_predict_returns_none():
    ws = _FakeWordSpace(_FakeDiscourse(predicted=None, confidence=None, bias=None))
    layer = SentencePrimingLayer(wordSpace_ref=lambda: ws, scale=0.1)
    x = torch.randn(2, 3, 4)
    y = layer(x)
    assert torch.equal(x, y)
