"""Tests confirming the sugar-absorption rules ``identity(S)`` and
``absorb(S, S)`` are wired through ``SyntacticLayer._RULE_METHODS`` and
behave as a unary pass-through and a binary left-pass respectively.

Plan reference: doc/plans/2026-04-26-rolling-cursor-doc-streaming-handoff.md §Verification
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch


def test_rule_methods_register_identity_and_absorb():
    """``_RULE_METHODS`` exposes both sugar-absorption methods."""
    from Language import SyntacticLayer
    table = SyntacticLayer._RULE_METHODS
    assert 'identity' in table, (
        "identity must be registered in _RULE_METHODS so the rule "
        "predictor can dispatch it"
    )
    assert 'absorb' in table, (
        "absorb must be registered in _RULE_METHODS so the rule "
        "predictor can dispatch it"
    )
    # identity is unary (binary=False); absorb is binary (binary=True).
    _, _, identity_binary = table['identity']
    _, _, absorb_binary = table['absorb']
    assert identity_binary is False, (
        f"identity must be a unary rule; got binary={identity_binary}"
    )
    assert absorb_binary is True, (
        f"absorb must be a binary rule; got binary={absorb_binary}"
    )
    # Neither has a reverse — they're lossy at the right operand.
    assert table['identity'][1] is None
    assert table['absorb'][1] is None


def test_identity_forward_returns_input_unchanged():
    """``identityForward`` is a pass-through: out == in."""
    from Language import SyntacticLayer, Grammar
    layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], grammar=Grammar())
    x = torch.randn(2, 4, 8)
    out = layer.identityForward(x, subspace=None)
    assert torch.equal(out, x), (
        "identityForward must return the input unchanged"
    )


def test_absorb_forward_returns_left_operand_only():
    """``absorbForward`` returns the left operand and ignores the right."""
    from Language import SyntacticLayer, Grammar
    layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], grammar=Grammar())
    left = torch.randn(2, 4, 8)
    right = torch.randn(2, 4, 8)   # arbitrary, must be discarded
    out = layer.absorbForward(left, right, subspace=None)
    assert torch.equal(out, left), (
        "absorbForward must return the left operand and discard the right"
    )


def test_absorb_forward_with_near_zero_right_still_returns_left():
    """Right operand near zero is the canonical sugar case; left wins."""
    from Language import SyntacticLayer, Grammar
    layer = SyntacticLayer(nInput=4, nOutput=4, rules=[], grammar=Grammar())
    left = torch.randn(2, 4, 8)
    right = torch.full_like(left, 1e-12)
    out = layer.absorbForward(left, right, subspace=None)
    # Output is bit-identical to left -- sugar absorption is lossless on left.
    assert torch.allclose(out, left, atol=0.0), (
        f"absorb on near-zero sugar must preserve left exactly; "
        f"max abs diff = {(out - left).abs().max().item()}"
    )


def test_grammar_inline_xml_with_sugar_rules_parses():
    """An inline XML with ``<S>identity(S)</S>`` and ``<S>absorb(S,S)</S>``
    parses to two new RuleDef entries with the expected method names."""
    import tempfile
    import textwrap
    import Language
    import util

    xml = textwrap.dedent("""\
        <?xml version="1.0" ?>
        <model>
          <WordSpace>
            <language>
              <start>S</start>
              <grammar>
                <S>not(S)</S>
                <S>identity(S)</S>
                <S>absorb(S, S)</S>
              </grammar>
            </language>
          </WordSpace>
        </model>
    """)
    with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False) as f:
        f.write(xml)
        path = f.name
    try:
        util.TheXMLConfig.load(path)
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()
        method_names = [r.method_name for r in Language.TheGrammar.rules]
        assert 'identity' in method_names, (
            f"identity rule must appear in the configured grammar; "
            f"got {method_names}"
        )
        assert 'absorb' in method_names, (
            f"absorb rule must appear in the configured grammar; "
            f"got {method_names}"
        )
    finally:
        Language.TheGrammar._configured = False
