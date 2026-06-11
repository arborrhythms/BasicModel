"""GrammarOpsPass §6d: the reference-partitioned codebook update law.

Author (2026-06-11): percepts are shaped by the PARALLEL pass; the
symbols (references) are shaped by the SERIAL pass — the serial pass
is the only one that does referential lookup, so it is the only one
that invokes the referential taxonomy qua references. STE is fine in
both cases: the real distinction is whether parallel mode may shape
references (it may NOT) and whether serial mode may shape
non-references (it may NOT).

Mechanism: ``Spaces.reference_update_mask`` (the law) +
``VectorQuantize.update_mask_fn`` (the chokepoint: EMA write,
accumulators, and dead-code expiry all respect it; byte-identical
legacy path when no mask is installed) +
``Space.install_reference_update_law`` (the wiring; lazy table getter;
dark when the meronomy is off).
"""
import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import VectorQuantize
from References import ReferenceTable
from Spaces import reference_update_mask, Space

V, D = 6, 4


def _knob(value):
    from util import TheXMLConfig
    if value is None:
        TheXMLConfig._data.get("architecture", {}).pop("meronomy", None)
    else:
        TheXMLConfig.set("architecture.meronomy", value)


def make_vq(seed=1, **kw):
    # Explicit CPU: byte-identity assertions require deterministic
    # index_add_, and a leaked MPS default-device MODE (see conftest's
    # device-leak note) makes EMA reductions nondeterministic AND
    # forces un-pinned constructors inside VQ.forward onto mps.
    # init_device("cpu") resets the global mode (the same guard the
    # XOR convergence fixture uses); the module-scoped conftest
    # fixture restores the prior device after this module.
    from util import init_device
    init_device("cpu")
    torch.manual_seed(seed)
    vq = VectorQuantize(dim=D, codebook_size=V, decay=0.5, **kw)
    vq.cpu()
    vq.train()
    return vq


def steps(vq, n=5, seed=2):
    torch.manual_seed(seed)
    for _ in range(n):
        vq(torch.randn(32, D, device="cpu"))


# ---------------------------------------------------------------------------
# The law itself.
# ---------------------------------------------------------------------------

def test_law_truth_table():
    refs = [1, 4]
    parallel = reference_update_mask(False, refs, V)
    serial = reference_update_mask(True, refs, V)
    assert parallel.tolist() == [True, False, True, True, False, True], (
        "parallel may not shape references")
    assert serial.tolist() == [False, True, False, False, True, False], (
        "serial may not shape non-references")
    assert torch.equal(parallel, ~serial), "exact partition, no gaps"


def test_law_edge_cases():
    assert reference_update_mask(False, [], V).all(), (
        "no references: parallel shapes everything (legacy)")
    assert not reference_update_mask(True, [], V).any(), (
        "no references: serial shapes nothing")
    # Out-of-range ids are ignored, not errors.
    m = reference_update_mask(False, [-1, 2, 99], V)
    assert m.tolist() == [True, True, False, True, True, True]


def test_table_accessors():
    t = ReferenceTable()
    t.bind(word=5, obj=2, licensed=True)
    t.bind(word=3, obj=2, licensed=True)   # synonym: same object
    t.bind(word=9, obj=0, licensed=True)
    assert t.bound_words() == [3, 5, 9]
    assert t.bound_objects() == [0, 2], "deduplicated object ids"


# ---------------------------------------------------------------------------
# The VQ chokepoint.
# ---------------------------------------------------------------------------

def test_no_mask_is_byte_identical():
    a = make_vq()
    b = make_vq()
    b.update_mask_fn = lambda V, device: None   # explicit None = legacy
    steps(a)
    steps(b)
    assert torch.equal(a.codebook, b.codebook)
    assert torch.equal(a.cluster_size, b.cluster_size)


def test_frozen_rows_do_not_move():
    frozen = torch.tensor([False, True, False, True, False, False])
    vq = make_vq()
    vq.update_mask_fn = lambda V, device: (~frozen).to(device)
    before_rows = vq.codebook.detach().clone()
    before_cs = vq.cluster_size.detach().clone()
    steps(vq)
    after_rows = vq.codebook.detach()
    after_cs = vq.cluster_size.detach()
    assert torch.equal(after_rows[frozen], before_rows[frozen]), (
        "frozen rows keep their codebook values exactly")
    assert torch.equal(after_cs[frozen], before_cs[frozen]), (
        "frozen rows keep their EMA accumulators (no decay, no mass) -- "
        "a later mask change cannot make them jump")
    assert not torch.equal(after_rows[~frozen], before_rows[~frozen]), (
        "allowed rows move")


def test_mask_flip_flips_the_partition():
    refs = [0, 2]
    is_ref = torch.zeros(V, dtype=torch.bool)
    is_ref[refs] = True
    # Parallel: refs frozen.
    vq_p = make_vq()
    vq_p.update_mask_fn = lambda Vn, device: (~is_ref).to(device)
    b_p = vq_p.codebook.detach().clone()
    steps(vq_p)
    assert torch.equal(vq_p.codebook.detach()[is_ref], b_p[is_ref])
    assert not torch.equal(vq_p.codebook.detach()[~is_ref], b_p[~is_ref])
    # Serial: non-refs frozen.
    vq_s = make_vq()
    vq_s.update_mask_fn = lambda Vn, device: is_ref.to(device)
    b_s = vq_s.codebook.detach().clone()
    steps(vq_s)
    assert torch.equal(vq_s.codebook.detach()[~is_ref], b_s[~is_ref])
    assert not torch.equal(vq_s.codebook.detach()[is_ref], b_s[is_ref])


def test_dead_code_revival_respects_the_freeze():
    frozen = torch.tensor([True, True, False, False, False, False])
    vq = make_vq(codebook_retire=True, threshold_ema_dead_code=1)
    vq.update_mask_fn = lambda Vn, device: (~frozen).to(device)
    before = vq.codebook.detach().clone()
    # Inputs clustered far from rows 0/1 so they would look dead.
    torch.manual_seed(3)
    for _ in range(8):
        vq(torch.randn(32, D, device="cpu") + 5.0)
    assert torch.equal(vq.codebook.detach()[frozen], before[frozen]), (
        "frozen rows are exempt from dead-code expiry/revival")


# ---------------------------------------------------------------------------
# The wiring: install_reference_update_law.
# ---------------------------------------------------------------------------

def _stub_space(vq):
    sp = Space.__new__(Space)
    cb = types.SimpleNamespace(vq=vq)
    sub = types.SimpleNamespace(codebook=lambda: cb)
    object.__setattr__(sp, 'subspace', sub)
    sp.serial_mode = False
    return sp


def test_install_and_dark_discipline():
    vq = make_vq()
    table = ReferenceTable()
    table.bind(word=1, obj=3, licensed=True)
    sp = _stub_space(vq)
    assert sp.install_reference_update_law(lambda: table, side='object')
    _knob(None)
    assert vq.update_mask_fn(V, None) is None, (
        "meronomy off: the law is dark (legacy behavior)")
    _knob("on")
    try:
        m = vq.update_mask_fn(V, None)
        expected = reference_update_mask(False, [3], V)
        assert torch.equal(m, expected), "parallel: bound object frozen"
        sp.serial_mode = True
        m = vq.update_mask_fn(V, None)
        assert torch.equal(m, reference_update_mask(True, [3], V)), (
            "serial: only the bound object moves")
    finally:
        _knob(None)


def test_install_word_side_and_no_table():
    vq = make_vq()
    sp = _stub_space(vq)
    holder = {'table': None}
    assert sp.install_reference_update_law(lambda: holder['table'],
                                           side='word')
    _knob("on")
    try:
        assert vq.update_mask_fn(V, None) is None, (
            "no table yet: legacy (the getter is lazy)")
        t = ReferenceTable()
        t.bind(word=2, obj=0, licensed=True)
        holder['table'] = t
        m = vq.update_mask_fn(V, None)
        assert torch.equal(m, reference_update_mask(False, [2], V)), (
            "word side: bound WORD ids are the references")
    finally:
        _knob(None)


def test_install_without_codebook_returns_false():
    sp = Space.__new__(Space)
    object.__setattr__(sp, 'subspace',
                       types.SimpleNamespace(codebook=lambda: None))
    assert sp.install_reference_update_law(lambda: None, side='object') \
        is False
