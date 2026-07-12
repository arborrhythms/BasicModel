"""Masked semantic reconstruction config (todo.md deliverable).

``MM_masked_semantic`` runs the whole-slab masked-LM (``create_ir_mask``
-> ``compute_masked``) on the PARALLEL sparse-concept path, so a masked
word is predicted with both the semantic-whole/category (META) evidence
minted under ``<mereologyRaise>`` and bottom-up attention over the
concept inventory (the FF pyramid). The ablation tests re-run the SAME
seeded build with one evidence source disabled and pin that the masked
predictions / the reverse reconstruction actually move -- the mask draws
are asserted identical across runs, so a diff is evidence flow, not RNG.

The role-PROFILE category learner (``observe_category_roles``) is fed
only by ``LanguageLayer.compose`` (the serial parser), so on this
parallel config the categoryCodebook VQ allocates (substrate roles) but
stays observation-idle; the live per-word category evidence here is the
META/type row of the word/object/meta triple ("wholes are types").

cpu/eager, seeded.
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import functools
import sys

import torch

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config

CFG = "data/MM_masked_semantic.xml"
# The META mint's reconstruction effect needs repeat presentations (the
# store rows minted on pass 1 are read on later passes).
EPOCHS = 3


def _run(ablate):
    """Fresh seeded build + EPOCHS single-batch epochs.

    Captures the masked-LM inputs and the reverse reconstruction at the
    two loss seams (``_masked_event_loss`` / ``_reverse_event_loss``).
    """
    torch.manual_seed(7)
    model, *_ = _build_model(_resolve_config(CFG))
    if ablate == "mint":
        # category/semantic-whole evidence OFF: no word/object/META triple.
        for ws in model.wholeSpaces:
            object.__setattr__(ws, "_mereology_raise", False)
        object.__setattr__(model.perceptualSpace, "_mereology_raise", False)
    elif ablate == "concept":
        # bottom-up concept attention OFF: sparse pump to the off-path.
        for cs in model.conceptualSpaces:
            object.__setattr__(cs, "_symbolic_order", 0)
    rec = {"masks": [], "losses": []}
    orig_m = model._masked_event_loss

    def spy_m(pred, target, mask, _o=orig_m, _r=rec):
        out = _o(pred, target, mask)
        _r["masks"].append(mask.detach().clone())
        _r["pred"] = pred.detach().clone()
        _r["pred_grad"] = bool(pred.requires_grad)
        _r["target"] = target.detach().clone()
        _r["losses"].append(float(out.detach()))
        return out

    model._masked_event_loss = spy_m
    orig_r = model._reverse_event_loss

    def spy_r(rev_ev, fwd_ev, _o=orig_r, _r=rec):
        _r["rev"] = rev_ev.detach().clone()
        return _o(rev_ev, fwd_ev)

    model._reverse_event_loss = spy_r
    opt = model.getOptimizer(lr=0.01)
    for e in range(EPOCHS):
        # Reseed per epoch: the sparse pump consumes RNG after the mask
        # draw, so a shared start seed alone lets later epochs' Bernoulli
        # masks drift between ablation runs.
        torch.manual_seed(1000 + e)
        model.runEpoch(optimizer=opt, batchSize=4, split="train",
                       max_batches=1)
    rec["model"] = model
    return rec


@functools.lru_cache(maxsize=None)
def _cached(key):
    return _run({"base": None, "base2": None,
                 "mint": "mint", "concept": "concept"}[key])


def _same_masks(a, b):
    return (len(a["masks"]) == len(b["masks"])
            and all(torch.equal(x, y)
                    for x, y in zip(a["masks"], b["masks"])))


def test_config_is_masked_parallel_semantic():
    """Structural pins: masked-IR training on the parallel sparse path."""
    m = _cached("base")["model"]
    assert m.mask_rate == 0.3
    assert m.serial is False and m.symbolicOrder == 3
    assert m.useGrammar == "none", (
        "an operator grammar would halve the tile taper and clip the "
        "whole-slab masked-LM compare")
    assert getattr(m, "symbol_tower", False)
    assert getattr(m, "mereology_raise", False)
    cs0 = m.conceptualSpaces[0]
    assert cs0._sparse_active()
    assert cs0._order_caps() == (8, 4, 2, 1), cs0._order_caps()


def test_masked_ir_training_engages():
    """Masked positions exist every epoch and contribute a live loss."""
    rec = _cached("base")
    assert rec["losses"], "masked-LM seam never reached"
    assert all(m.sum() > 0 for m in rec["masks"]), "no masked positions"
    assert all(l > 0.0 for l in rec["losses"]), rec["losses"]
    assert rec["pred_grad"], "masked prediction must carry gradient"
    assert rec["model"]._d3_active is False, (
        "D3 per-word objective must not displace the whole-slab masked-LM")
    # The body in-fills: pred at masked positions is neither the zeroed
    # MASK content nor the target.
    mask = rec["masks"][-1].bool()
    sub = rec["model"].perceptualSpace.subspace
    nWhat = rec["pred"].shape[-1] - int(sub.nWhere) - int(sub.nWhen)
    pm = rec["pred"][mask][..., :nWhat]
    tm = rec["target"][mask][..., :nWhat]
    assert float(pm.abs().max()) > 0.0
    assert not torch.allclose(pm, tm)


def test_concept_attention_live_not_dark():
    """The FF pyramid populates per-rung stats; the category VQ allocates."""
    m = _cached("base")["model"]
    cs0 = m.conceptualSpaces[0]
    lv = getattr(cs0, "_cs_level_acts", None)
    assert lv is not None and float(lv[0]) > 0.0, lv
    ws = m.wholeSpaces[-1]
    assert ws.category_codebook_enabled()
    assert int(getattr(ws, "_category_n_roles", 0)) > 0


def test_masked_prediction_uses_concept_attention():
    """Disabling the sparse pump moves the masked predictions AND the
    reverse reconstruction (same mask draws -> the diff is evidence)."""
    base, con = _cached("base"), _cached("concept")
    assert _same_masks(base, con), "mask draws diverged: RNG confound"
    assert float((base["pred"] - con["pred"]).abs().max()) > 1e-3
    assert float((base["rev"] - con["rev"]).abs().max()) > 1e-3


def test_reconstruction_uses_category_evidence():
    """Disabling the word/object/META mint moves the reverse
    reconstruction (the masked word's decode path)."""
    base, mint = _cached("base"), _cached("mint")
    assert _same_masks(base, mint), "mask draws diverged: RNG confound"
    assert float((base["rev"] - mint["rev"]).abs().max()) > 1e-3


def test_seeded_rerun_is_deterministic():
    """The ablation thresholds sit far above the rerun noise floor."""
    a, b = _cached("base"), _cached("base2")
    assert _same_masks(a, b)
    assert float((a["pred"] - b["pred"]).abs().max()) < 1e-4
    assert float((a["rev"] - b["rev"]).abs().max()) < 1e-4
