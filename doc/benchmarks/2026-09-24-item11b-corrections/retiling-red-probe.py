import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from Layers import RadixLayer

def test_definition_group_tracks_new_overlapping_canonical_parts():
    store = RadixLayer(4, initial_cap=32, promotion_threshold=2)
    a, b, c = store.spell_out(b"abc")
    store.begin_turn()
    store.observe_chunk(b"bc")
    bc = store.observe_chunk(b"bc")
    old_literal = store.canonical_parts([a, b, c])
    assert old_literal == [a, bc]
    store.begin_turn()
    store.observe_chunk(b"ab")
    ab = store.observe_chunk(b"ab")
    assert store.spell_out(b"abc") == [ab, c]
    assert store.canonical_parts(old_literal) == [ab, c]
