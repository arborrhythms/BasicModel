"""Read-only replay/recall diagnosis; no seed, objective or threshold changes."""
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import torch
from util import init_device
from What import What
from test_output_walk import _model, _capture_program_probe
from test_compiled_word_chunk import _stage_fullgraph_tensor_peer
from test_reverse_traversal import _stage_packed
from test_meronomy_ladder import _build_ladder_variant

init_device('cpu')
m = _model()
m._tensor_peer_while_eager = True
m._chart_compose_per_word = lambda: None
with torch.no_grad():
    _stage_fullgraph_tensor_peer(m, ['12 plus 1', '3 plus 4'])
    out = m._forward_with_compiled_sentence_state(None)
    m._publish_compiled_sentence_state(out)
    u = m._capture_understanding(out[:4])
    d = m._resolve_answer(u, What.supervised(0))
    idea, *_ = m._materialize_answer_idea(u, d, What.supervised(0))
    end = m._sentence_end_state(None)
    replay, _ = m._materialize_entries(u.answer_program, torch.zeros_like(end), m._walk_budget())
    print('IDEA_END_DIFF', (idea - end).abs().amax(-1).tolist(), flush=True)
    print('REPLAY_END_DIFF', (replay - end).abs().amax(-1).tolist(), flush=True)
    print('IDEA_REPLAY_DIFF', (idea - replay).abs().amax(-1).tolist(), flush=True)
    print('DEPTH', m.conceptualSpace.stm._depth, flush=True)
    for b, p in enumerate(u.answer_program):
        print('PROGRAM', b, p.actions.tolist(), flush=True)
    trace = m._reconstruction_stack()
    for name in ('_choice_positions', '_choice_actions', '_choice_mask'):
        print(name, getattr(trace, name, None), flush=True)
m.End(); m.symbolSpace.soft_reset()

m = _build_ladder_variant(Path(tempfile.mkdtemp()), 'probe_packed', [
    ('<serialWordCapacity>8</serialWordCapacity>', '<serialWordCapacity>16</serialWordCapacity>'),
    ('<serialWordBuckets>8</serialWordBuckets>', '<serialWordBuckets>16</serialWordBuckets>'),
    ('<training>', '<training>\n      <outputInLoop>true</outputInLoop>'),
    ('<sentenceExpectation>false</sentenceExpectation>', '<sentenceExpectation>true</sentenceExpectation>')])
m._tensor_peer_while_eager = True
m._chart_compose_per_word = lambda: None
m._install_unit_span_fn()
with torch.no_grad():
    _stage_packed(m, [['12 plus 1', '3 plus 4'], ['8 plus 2']])
    out = m._forward_with_compiled_sentence_state(None)
    u = m._capture_understanding(out)
    roots = out[7].clone()
    valid = m.inputSpace._packed_sentence_slot_mask.clone()
    print('PACK_VALID', valid.tolist(), flush=True)
    print('PACK_DEPTHS', m._tensor_sentence_depths.tolist(), flush=True)
    print('BEFORE_HISTORY', {b:len(v) for b,v in m._recall_program_history().items()}, flush=True)
    _capture_program_probe(m, ['5 plus 9', '4 plus 1'])
    print('RESTAGED_HISTORY', {b:len(v) for b,v in m._recall_program_history().items()}, flush=True)
    for t in (0, 1):
        m._observe_discourse(m.symbolSpace.discourse, roots[:,t], mask=valid[:,t], slot=t, understanding=u)
        print('OBSERVED', t, {b:len(v) for b,v in m._recall_program_history().items()}, flush=True)
m.End(); m.symbolSpace.soft_reset()
