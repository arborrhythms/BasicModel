"""Compiled reverse-loops plan, slice 3: the answer-side generate walk as
the second compiled loop (doc/plans/2026-09-12-compiled-reverse-loops.md)."""
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "bin"), str(_ROOT / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_meronomy_ladder import _build_ladder  # noqa: E402


def _model():
    m = _build_ladder()
    m.output_in_loop = True
    return m


def _stamped_event(m, rule_name, N=4):
    """A ``[1, N, D]`` event whose top live slot is stamped with the
    global rule id of ``rule_name``; slots below it are terminals."""
    import Language
    language = m.languageSpace
    binary = language._tree_layer(2)
    names = list(binary.op_names)
    idx = names.index(rule_name)
    global_id = int(language._cs_binary_rule_ids[idx])
    gl = getattr(list(binary.ops)[idx], "gl", list(binary.ops)[idx])
    cw = int(gl._content_width)
    D = int(m.conceptualSpace.stm.concept_dim)
    torch.manual_seed(0)
    event = torch.zeros(1, N, D)
    live = 2
    event[0, :live, :cw] = torch.tanh(torch.randn(live, cw) * 0.5)
    event[0, 0, cw] = 0.0                                                   # a plain concept slot
    event[0, 1, cw] = float(Language.TheGrammar.where_id_for_rule(global_id))  # the rule on top
    return event, gl, cw, live


def test_walk_unreduces_a_rule_stamped_top_with_the_tied_inverse():
    m = _model()
    for rule in ("lift", "lower"):
        names = list(m.languageSpace._tree_layer(2).op_names)
        if rule not in names:
            continue
        event, gl, cw, live = _stamped_event(m, rule)
        out, n_live, truncated = m._output_generate_walk(event, budget=3)
        assert n_live.tolist() == [live + 1]
        assert not bool(truncated.any())
        parent = event[0, 1]
        left, right = out[0, 1], out[0, 2]
        assert float(left[cw]) == 0.0 and float(right[cw]) == 0.0      # children stamped empty
        recomposed = gl.compose(left.unsqueeze(0), right.unsqueeze(0))[0]
        assert torch.allclose(recomposed[:cw], parent[:cw], atol=1e-3), rule
        assert torch.equal(out[0, 0], event[0, 0])                       # the terminal untouched


def test_walk_leaves_a_terminal_top_alone_and_reports_no_truncation():
    m = _model()
    event, gl, cw, live = _stamped_event(m, "lift")
    event[0, 1, cw] = 0.0                                   # no rule stamp on top
    out, n_live, truncated = m._output_generate_walk(event, budget=3)
    assert torch.equal(out, event) and n_live.tolist() == [live]
    assert not bool(truncated.any())


def test_walk_reports_truncation_when_no_slot_is_free():
    m = _model()
    event, gl, cw, live = _stamped_event(m, "lift", N=2)
    out, n_live, truncated = m._output_generate_walk(event, budget=3)
    assert bool(truncated.all()) and n_live.tolist() == [live]


def test_walk_compiles_fullgraph_and_matches_eager():
    m = _model()
    event, gl, cw, live = _stamped_event(m, "lift")
    eager = m._output_generate_walk(event, budget=3)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(m._output_generate_walk, backend="eager", fullgraph=True)
    try:
        out, n_live, truncated = compiled(event, 3)
        assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1
        assert torch.allclose(out, eager[0], atol=1e-6)
        assert torch.equal(n_live, eager[1]) and torch.equal(truncated, eager[2])
    finally:
        torch._dynamo.reset()


def test_walk_handles_mixed_output_lengths_per_row():
    """Output gate: one row un-reduces (three live slots), the other keeps
    its terminal; per-row live counts differ and neither row truncates."""
    m = _model()
    e0, gl, cw, live = _stamped_event(m, "lift")
    e1 = e0.clone()
    e1[0, 1, cw] = 0.0                                      # row 1: terminal on top
    event = torch.cat((e0, e1), dim=0)
    out, n_live, truncated = m._output_generate_walk(event, budget=3)
    assert n_live.tolist() == [live + 1, live]
    assert not bool(truncated.any())
    assert torch.equal(out[1], event[1])
    assert float(out[0, 2].abs().sum()) > 0.0               # row 0 opened a slot


def test_walk_is_invariant_to_reconstruction_only_state():
    """Output gate: the walk reads nothing of the input reconstruction
    (contract 5): changing the reconstruction results leaves it unchanged."""
    m = _model()
    event, gl, cw, live = _stamped_event(m, "lift")
    base = m._output_generate_walk(event, budget=3)
    D = int(m.conceptualSpace.stm.concept_dim)
    object.__setattr__(m, "_recon_ideas", torch.randn(1, 4, D))
    object.__setattr__(m, "_recon_cost", torch.tensor([7.0]))
    object.__setattr__(m, "_recon_truncated", torch.tensor([True]))
    again = m._output_generate_walk(event, budget=3)
    assert torch.equal(again[0], base[0])
    assert torch.equal(again[1], base[1]) and torch.equal(again[2], base[2])
