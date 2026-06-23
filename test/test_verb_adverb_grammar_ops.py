"""Live grammar operators for verb spectra and adverb eigenmodifiers."""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_ROOT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Language import (
    AdverbLayer,
    Grammar,
    GRAMMAR_LAYER_CLASSES,
    VerbLayer,
)

_D = 8


def _np_and_word(seed=0):
    g = torch.Generator().manual_seed(seed)
    np_ = torch.randn(2, 5, _D, generator=g).tanh() * 0.6
    word = torch.randn(2, 5, _D, generator=g).tanh() * 0.6
    return np_, word


def _strong_spectrum(layer, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer._verb_spec.weight.copy_(
            torch.randn(layer._verb_spec.weight.shape, generator=g) * 0.6)
        layer._verb_spec.bias.fill_(0.3)
    return layer


def _strong_adverb(layer, seed=2):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer._adv_edit.weight.copy_(
            torch.randn(layer._adv_edit.weight.shape, generator=g) * 0.8)
        layer._adv_edit.bias.fill_(0.4)
    return layer


def _methods(fname):
    g = Grammar()
    g.load_from_grammar_file(fname)
    up = {r.method_name for r in g.rules_upward if r.method_name}
    down = {r.method_name for r in g.rules_downward if r.method_name}
    return up, down


def test_verb_and_adverb_registered():
    assert GRAMMAR_LAYER_CLASSES["verb"] is VerbLayer
    assert GRAMMAR_LAYER_CLASSES["adverb"] is AdverbLayer


def test_verb_layer_forces_spectrum_and_inverts_given_verb():
    layer = VerbLayer(nInput=_D, nOutput=_D)
    assert layer._verb_spectrum is True
    assert layer._verb_spec is not None
    np_, verb = _np_and_word()
    torch.testing.assert_close(layer.forward(np_, verb), np_, atol=1e-5, rtol=1e-5)

    layer = _strong_spectrum(VerbLayer(nInput=_D, nOutput=_D))
    vp = layer.forward(np_, verb)
    assert float((vp - np_).abs().max()) > 0.05
    recovered, returned_verb = layer.reverse(vp, verb_what=verb)
    torch.testing.assert_close(recovered, np_, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(returned_verb, verb)


def test_adverb_layer_forces_edit_and_is_zero_init_noop():
    layer = AdverbLayer(nInput=_D, nOutput=_D)
    assert layer._adverb_eig_edit is True
    assert layer._adv_edit is not None
    vp, adv = _np_and_word()
    torch.testing.assert_close(layer.forward(vp, adv), vp, atol=1e-5, rtol=1e-5)

    layer = _strong_adverb(AdverbLayer(nInput=_D, nOutput=_D))
    edited = layer.forward(vp, adv)
    assert float((edited - vp).abs().max()) > 0.1


def test_default_and_complete_have_verb_adverb_not_shamatha():
    for fname in ("default.grammar", "complete.grammar"):
        up, down = _methods(fname)
        assert {"verb", "adverb"} <= up, fname
        assert {"verb", "adverb"} <= down, fname

    up, down = _methods("shamatha.grammar")
    assert "verb" not in up and "adverb" not in up
    assert "verb" not in down and "adverb" not in down


def test_default_grammar_replaces_lift_with_verb():
    up, down = _methods("default.grammar")
    assert "verb" in up and "verb" in down
    assert "lift" not in up and "lift" not in down


def test_complete_contains_absorbed_role_only_methods():
    complete_up, complete_down = _methods("complete.grammar")
    absorbed = {"preposition", "bind", "tense", "morphology"}
    assert absorbed <= complete_up, absorbed - complete_up
    assert absorbed <= complete_down, absorbed - complete_down
