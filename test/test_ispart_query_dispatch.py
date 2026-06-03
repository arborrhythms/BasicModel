"""isPart relation + query-based dispatch (Phase R1.2).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
decision 6 + §6: ``isEqual`` and ``isPart`` are each a *single* grammar
relation; ``query="true"`` selects answer-producing semantics,
``query="false"`` selects assertive semantics. This folds in (replaces)
the separate ``queryPart`` / ``assertPart`` operators. ``isPart`` is the
S-tier assertive parthood relation -- the mereological analogue of the
S-tier assertive ``isEqual`` -- and dispatches to the existing
parthood-truth ``queryPart`` layer when interrogative.
"""

import os
import sys
import textwrap

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def _ispart_grammar(query_attr):
    return textwrap.dedent("""\
        <?xml version="1.0"?>
        <grammar name="ispart_probe">
          <SymbolicSpace>
            <start name="relative_truth">isPart_O1</start>
            <compose>
              <rule query="%s">isPart_O1 = isPart.forward(isPart_I1, isPart_I2)</rule>
            </compose>
            <generate>
              <rule query="%s">isPart_I1, isPart_I2 = isPart.reverse(isPart_O1)</rule>
            </generate>
          </SymbolicSpace>
        </grammar>
    """ % (query_attr, query_attr))


def _load(text, monkeypatch, tmp_path):
    import Language
    path = tmp_path / "probe.grammar"
    path.write_text(text)
    monkeypatch.setattr(Language, "_GRAMMAR_DIR", tmp_path)
    g = Language.Grammar()
    g.load_from_grammar_file("probe.grammar")
    return g


def test_ispart_layer_registered():
    """``isPart`` is in the layer registry as an arity-2, S-tier relation."""
    from Language import GRAMMAR_LAYER_CLASSES
    assert "isPart" in GRAMMAR_LAYER_CLASSES
    cls = GRAMMAR_LAYER_CLASSES["isPart"]
    assert cls.rule_name == "isPart"
    assert cls.arity == 2
    assert cls.tier == "S", "isPart is the S-tier assertive relative-truth relation"


def test_ispart_assertive_forward_passes_parent():
    """Assertive ``isPart(A, B)`` yields the encompassing parent ``B`` (like
    the C-tier ``part``), with the lossy ``(parent, parent)`` pseudo-inverse."""
    from Language import GRAMMAR_LAYER_CLASSES
    layer = GRAMMAR_LAYER_CLASSES["isPart"]()
    left = torch.randn(2, 4)
    right = torch.randn(2, 4)
    out = layer.forward(left, right)
    assert torch.equal(out, right)
    lo, ro = layer.reverse(out)
    assert torch.equal(lo, out) and torch.equal(ro, out)


def test_ispart_query_false_dispatches_assertive(monkeypatch, tmp_path):
    """``query="false"`` keeps the assertive ``isPart`` layer."""
    from Language import _dispatch_method_name_for_rule
    g = _load(_ispart_grammar("false"), monkeypatch, tmp_path)
    rule = next(r for r in g.rules_upward if r.method_name == "isPart")
    assert rule.query is False
    assert _dispatch_method_name_for_rule(rule) == "isPart"


def test_ispart_query_true_dispatches_to_query_part(monkeypatch, tmp_path):
    """``query="true"`` selects the answer-producing ``queryPart`` layer."""
    from Language import _dispatch_method_name_for_rule
    g = _load(_ispart_grammar("true"), monkeypatch, tmp_path)
    rule = next(r for r in g.rules_upward if r.method_name == "isPart")
    assert rule.query is True
    assert _dispatch_method_name_for_rule(rule) == "queryPart"
