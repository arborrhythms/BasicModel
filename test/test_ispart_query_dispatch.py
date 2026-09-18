"""``part`` is one pure structural face; thought runs only at a boundary."""

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


def _part_grammar(query_attr=None):
    attr = '' if query_attr is None else f' query="{query_attr}"'
    return textwrap.dedent("""\
        <?xml version="1.0"?>
        <grammar name="part_probe">
          <Symbolic>
            <start name="relative_truth">part_O1</start>
            <compose>
              <rule%s>part_O1 = part.forward(part_I1, part_I2)</rule>
            </compose>
            <generate>
              <rule%s>part_I1, part_I2 = part.reverse(part_O1)</rule>
            </generate>
          </Symbolic>
        </grammar>
    """ % (attr, attr))


def _load(text, monkeypatch, tmp_path):
    import Language
    path = tmp_path / "probe.grammar"
    path.write_text(text)
    monkeypatch.setattr(Language, "_GRAMMAR_DIR", tmp_path)
    g = Language.Grammar()
    g.load_from_grammar_file("probe.grammar")
    return g


def test_part_layer_registered():
    """``part`` is the arity-2 conceptual structural relation."""
    from Language import GRAMMAR_LAYER_CLASSES
    assert "part" in GRAMMAR_LAYER_CLASSES
    cls = GRAMMAR_LAYER_CLASSES["part"]
    assert cls.rule_name == "part"
    assert cls.arity == 2
    assert cls.space_role == "CS"


def test_part_structural_forward_passes_parent():
    """Structural ``part(A, B)`` yields its whole without answering it."""
    import pytest
    from Language import GRAMMAR_LAYER_CLASSES
    layer = GRAMMAR_LAYER_CLASSES["part"]()
    left = torch.randn(2, 4)
    right = torch.randn(2, 4)
    out = layer.forward(left, right)
    assert torch.equal(out, right)
    with pytest.raises(NotImplementedError, match="part"):
        layer.reverse(out)


def test_part_dispatches_its_declared_pure_structural_face(monkeypatch, tmp_path):
    """The structural spelling determines dispatch; mode is not a rule attribute."""
    from Language import _dispatch_method_name_for_rule
    g = _load(_part_grammar(), monkeypatch, tmp_path)
    rule = next(r for r in g.rules_upward if r.method_name == "part")
    assert _dispatch_method_name_for_rule(rule) == "part"


def test_retired_rule_query_attribute_fails_loudly(monkeypatch, tmp_path):
    import pytest
    with pytest.raises(ValueError, match="query attribute|retired"):
        _load(_part_grammar("true"), monkeypatch, tmp_path)
