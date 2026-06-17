"""Live E/M smoke for the MetaSymbol Category codebook (handoff 2026-06-15 §3.2).

The Phase-1 codebook mechanics are unit-proven (test_category_codebook.py) and
the perception->role_obs->E/M chain is wired, but it had not been exercised on a
REAL model. This stands up a tiny BasicModel (POS_smoke.xml's inline word corpus
+ operator grammar, so compute_role_vocabulary yields roles > 0) with the
category codebook enabled, runs a handful of perception+compose forwards, and
asserts the chain actually fires end-to-end:

  * the codebook lazily enables (roles > 0) from the autobind hook;
  * the autobind stashes per-position percept ids (_category_last_pid) and
    records MetaSymbol -> centroid assignments (_category_assign);
  * centroid role vectors populate (the role M-step ran).

Phase 2 (chooser conditioning): with transformChooser=mlp +
chooserCategoryContext, the structured layers' MLP chooser is sized with the
role-count context block and compose builds a [B, N, n_roles] context.
"""

import os
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_ROOT, "bin")
_DATA = os.path.join(_ROOT, "data")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import pytest

import Models
import Language
from util import init_config, TheXMLConfig
from Language import compute_role_vocabulary

_BASE_CONFIG = os.path.join(_DATA, "POS_smoke.xml")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _write_category_config(*, mlp=False, chooser_ctx=False):
    """Write a temp config = POS_smoke.xml + the category flags, injected in
    xsd-sequence order right after <symbolicOrder>. Forces PARALLEL mode
    (symbolicOrder=0): the round-0 role observation that drives the E/M is
    parallel-mode-correct (serial per-word attribution is approximate -- the
    handoff §3.3 caveat). Written into data/ at runtime (invisible to the
    import-time data/*.xml sweeps) and unlinked by the caller."""
    with open(_BASE_CONFIG) as fh:
        text = fh.read()
    flags = "    <categoryCodebook>true</categoryCodebook>\n"
    if mlp:
        # transformChooser precedes categoryCodebook in the xsd sequence.
        flags = "    <transformChooser>mlp</transformChooser>\n" + flags
    if chooser_ctx:
        flags = flags + "    <chooserCategoryContext>true</chooserCategoryContext>\n"
    needle = "<symbolicOrder>1</symbolicOrder>"
    assert needle in text, "POS_smoke.xml shape changed"
    text = text.replace(
        needle, "<symbolicOrder>0</symbolicOrder>\n" + flags, 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA)
    tmp.write(text)
    tmp.close()
    return tmp.name


def _build(path):
    init_config(path=path, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    dat = TheXMLConfig.data.get("architecture", {}).get("data", {})
    Models.TheData.load("inline", dat=dat)
    model, _ = Models.BasicModel.from_config(path)
    return model


def _run_forwards(model, n=15):
    n_train = len(Models.TheData.train_input)
    loader = model.inputSpace.data.data_loader(
        split="train", num_streams=n_train)
    model.train()
    fired = 0
    cs = model.conceptualSpace
    with torch.no_grad():
        for _ in range(n):
            inp_items, _out = next(iter(loader))
            ipt = model.inputSpace.prepInput(inp_items)
            if ipt is None:
                continue
            model.forward(ipt)
            # The autobind + category E/M is committed at the sentence-boundary
            # hard Reset (off the compiled forward), exactly as the training
            # loop drives it. The bare forward alone does not fire it.
            cs.Reset(hard=True)
            fired += 1
    return fired


def _terminal_ss(model):
    """The WholeSpace the autobind hook targets (terminalSymbolicSpace_ref when
    wired, else the model's symbolicSpace)."""
    cs = getattr(model, "conceptualSpace", None)
    ss = getattr(cs, "terminalSymbolicSpace_ref", None) if cs is not None else None
    return ss if ss is not None else getattr(model, "symbolicSpace", None)


def test_codebook_enables_and_em_populates_on_real_model():
    path = _write_category_config()
    try:
        model = _build(path)
        # Roles exist for this grammar (operator methods -> op_I/op_O roles).
        _r, _i, n_roles = compute_role_vocabulary(Language.TheGrammar)
        assert n_roles > 0, "operator grammar should declare roles"
        fired = _run_forwards(model)
        assert fired > 0, "no forward pass completed"
        ss = _terminal_ss(model)
        assert ss is not None
        # The autobind hook lazily enabled the codebook with the grammar roles.
        assert ss.category_codebook_enabled(), "codebook never enabled"
        assert int(ss._category_n_roles) == n_roles
        # Phase 2 stash: per-position percept ids recorded for compose.
        assert getattr(ss, "_category_last_pid", None) is not None
        # The E-step ran: at least one MetaSymbol assigned to a centroid.
        assert len(getattr(ss, "_category_assign", {})) > 0, (
            "no MetaSymbol -> centroid assignment recorded")
        # The role M-step ran: at least one centroid role vector is non-zero.
        assert float(ss._category_role.abs().sum()) > 0.0, (
            "centroid role vectors never populated")
    finally:
        os.unlink(path)


def test_phase2_chooser_sized_and_context_built():
    path = _write_category_config(mlp=True, chooser_ctx=True)
    try:
        model = _build(path)
        _r, _i, n_roles = compute_role_vocabulary(Language.TheGrammar)
        assert n_roles > 0
        # The router's structured-layer MLP choosers were sized with the
        # category context block (width == role count).
        router = model.wordSubSpace.languageLayer
        sized = []
        for layers in (router._unary_layers, router._binary_layers):
            for layer in layers.values():
                ch = getattr(layer, "chooser", None)
                nrc = getattr(ch, "n_role_cats", None)
                if nrc is not None:
                    sized.append(int(nrc))
        assert sized, "no structured-layer chooser found"
        assert all(s == n_roles for s in sized), (sized, n_roles)
        # Drive perception so the codebook enables + assignments populate, then
        # the compose-side context builder returns [B, N, n_roles].
        _run_forwards(model)
        ss = _terminal_ss(model)
        assert ss.category_codebook_enabled()
        x = torch.zeros(2, 4, int(ss.nDim))
        cat = router._build_category_context(x, ss)
        # None is acceptable only before any percept is bound; after forwards a
        # stash exists, so we expect a populated [B, N, n_roles] tensor.
        assert cat is not None
        assert tuple(cat.shape) == (2, 4, n_roles)
    finally:
        os.unlink(path)
