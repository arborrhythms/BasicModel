"""Radix/meronomy reconstruction RENDER wiring (2026-07-02 plan C2/B2).

The reverse-decode machinery existed end-to-end (RadixLayer.reverse is the
structural decode; PartSpace.reconstruct_to_buffer the renderer) but the
staging only fired on the Embedding (lexicon) branch of PartSpace.reverse --
radix/meronomy configs raised "reconstruct_to_buffer() called before
reverse()". These tests pin the wiring: the numeric reverse stages a radix
render thunk, and rendering decodes known chunks back to their bytes.
"""
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Language

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_radix_render_decodes_known_chunks_consecutively():
    """Unit: a staged radix thunk renders known chunk rows to their bytes
    (offsets absent -> consecutive placement, same as the Embedding path)."""
    m = _build("MM_20M_xor.xml")
    ps = m.perceptualSpace
    radix = ps.vocabulary
    pid_a = radix.insert(b"hello")
    pid_b = radix.insert(b" world")
    D = int(radix.codebook.shape[-1])
    # Real muxed width: content in the leading D dims, where/when band zeros
    # (zero sin/cos -> offset None -> consecutive placement).
    event = torch.zeros(1, 2, int(ps.muxedSize))
    event[0, 0, :D] = radix.vector_for(pid_a).detach()
    event[0, 1, :D] = radix.vector_for(pid_b).detach()
    object.__setattr__(ps, "_recovered_input", None)
    object.__setattr__(ps, "_recovered_input_thunk",
                       ("radix", radix, event, ps.subspace))
    out = ps.reconstruct_to_buffer(buf_size=32)
    assert out == ["hello world"]
    # reconstruct_data + get_recovered_word ride the same meta.
    assert ps.get_recovered_word(0, 0) == "hello"
    words = ps.reconstruct_data()
    assert words[0] == ["hello", " world"]


def test_reconstruction_seed_is_live_in_parallel_train():
    """The recon loss was a CONSTANT (grad-dead): the reverse was seeded from
    the detached STM snapshot and the chain is only input-differentiable.
    In parallel train mode the seed must be the LIVE stage carrier."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_20M_xor.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.train()
    m.forward(x)
    seed = m._reconstruction_seed()
    assert seed is not None and seed.dim() == 3
    assert seed.requires_grad                        # LIVE, trains the forward
    # And the reverse transmits that gradient end-to-end into lossRev.
    cs = m.conceptualSpace
    cs.subspace.set_event(seed)
    rev = m.reverse(cs.subspace)
    ev = rev.materialize() if rev is not None else None
    assert ev is not None and ev.requires_grad


def test_radix_full_reverse_stages_the_render():
    """Integration: forward + model reverse on the radix config STAGES the
    render -- reconstruct_to_buffer returns strings instead of raising
    'called before reverse()' (the B2 symptom)."""
    import Models
    from util import TheXMLConfig
    m = _build("MM_20M_xor.xml")
    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(items)
    m.eval()
    with torch.no_grad():
        m.forward(x)
        m.reverse(getattr(m, "_combine_last_cs_sub", None)
                  or m.conceptualSpace.subspace)
    out = m.perceptualSpace.reconstruct_to_buffer(buf_size=64)
    assert isinstance(out, list) and len(out) == len(items)
    assert all(isinstance(s, str) for s in out)
