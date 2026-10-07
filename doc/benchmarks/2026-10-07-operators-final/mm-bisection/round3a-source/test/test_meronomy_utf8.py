"""Exact byte-witness ports from the retired standalone perceptual analyzer."""
import pytest
import torch

from test_meronomy_ladder import _stage


def _new_ladder(tmp_path):
    # No training or RNG seeding. Sixteen unit slots fit the original surfaces.
    import Language
    import Models
    from data import TheData
    from util import init_config
    from pathlib import Path
    data_dir = Path(__file__).resolve().parents[1] / "data"
    source = (data_dir / "MM_ladder.xml").read_text()
    for name in ("serialWordCapacity", "serialWordBuckets"):
        source = source.replace(f"<{name}>8</{name}>", f"<{name}>16</{name}>")
    config = tmp_path / "MM_ladder_byte_witness.xml"
    config.write_text(source)
    init_config(path=str(config), defaults_path=str(data_dir / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(str(config))
    TheData.load("math", dat=cfg["architecture"]["data"])
    model, _ = Models.BaseModel.from_config(str(config), data=TheData)
    model = model.to("cpu")
    return model


@pytest.fixture(scope="module")
def native_ladder(tmp_path_factory):
    model = _new_ladder(tmp_path_factory.mktemp("byte-witness"))
    yield model
    model.End()




def _witness(model, surface):
    # Test the eager byte boundary, independent of a dataset string adapter.
    slab = torch.tensor(list(surface.encode("utf-8")), dtype=torch.long)
    units, atoms, ids, mask, offsets = _stage(model, [slab], byte_witness=True)
    raw = b"".join(b"".join(unit) for unit in atoms[0])
    spans = model.perceptualSpace._forward_input["native_part_spans"][0]
    live_spans = [tuple(span) for span in spans.tolist() if span[1] > span[0]]
    return raw, live_spans, units[0], atoms[0], ids, mask


def test_byte_witness_round_trips_non_ascii(native_ladder):
    for surface in ("café déjà", "naïve—oz", "日本語 test", "aéb"):
        raw, *_ = _witness(native_ladder, surface)
        assert raw.decode("utf-8") == surface, surface


def test_byte_witness_offsets_cover_the_surface_without_overlap(native_ladder):
    surface = "café déjà"
    raw, spans, *_ = _witness(native_ladder, surface)
    spans = sorted(spans)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(surface.encode("utf-8"))
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert e0 == s1, spans
    assert raw == surface.encode("utf-8")


def test_split_multibyte_witness_keeps_raw_bytes(native_ladder):
    raw, *_ = _witness(native_ladder, "é")
    assert "�" not in raw.decode("utf-8")
    assert raw == "é".encode("utf-8")




@pytest.mark.parametrize("surface", ["the book has a cover", "a b", "book xyz"])
def test_surface_witness_replays_words_and_spaces(native_ladder, surface):
    raw, spans, units, atoms, *_ = _witness(native_ladder, surface)
    assert raw.decode("utf-8") == surface
    assert " " in raw.decode("utf-8")
    if surface == "book xyz":
        assert "book" in units


def test_unfamiliar_word_keeps_every_nonzero_byte_row(native_ladder):
    model = native_ladder
    raw, spans, units, atoms, ids, mask = _witness(model, "zq")
    assert raw == b"zq"
    assert spans == [(0, 1), (1, 2)]
    rows = model.perceptualSpace.percept_store._basis.lookup_rows(ids[mask])
    assert rows.shape[0] == 2
    assert bool((torch.linalg.vector_norm(rows, dim=-1) > 0).all())
