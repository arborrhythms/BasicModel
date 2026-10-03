"""Packing must preserve each sentence's ended state and tied inverse.

The native probe uses ambient model initialization. Vocabulary is admitted
once before either layout; no optimizer or dictionary rotation occurs.
"""
from pathlib import Path

import pytest
import torch


ATOL, RTOL = 1e-6, 1e-5


def build_model(tmp_path, *, active_vectors=None, word_capacity=32):
    import Language
    from Models import BaseModel
    from data import TheData
    from util import init_config, init_device

    root = Path(__file__).resolve().parents[1]
    source = (root / "data/MM_ladder.xml").read_text()
    if active_vectors is not None:
        source = source.replace("<activeVectors>4096</activeVectors>",
                                f"<activeVectors>{active_vectors}</activeVectors>", 1)
    for old, new in (
        ("<serialWordCapacity>8</serialWordCapacity>",
         f"<serialWordCapacity>{word_capacity}</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>",
         f"<serialWordBuckets>{word_capacity}</serialWordBuckets>"),
        ("<training>", "<training><reconstructInLoop>true</reconstructInLoop>"),
    ):
        assert old in source
        source = source.replace(old, new, 1)
    config = tmp_path / "packed_parity.xml"
    config.write_text(source)
    init_device("cpu")
    init_config(str(config), defaults_path=str(root / "data/model.xml"))
    cfg = BaseModel.load_config(str(config))
    TheData.load(cfg["architecture"]["data"]["dataset"],
                 dat=dict(cfg["architecture"]["data"]))
    Language.TheGrammar._configured = False
    model, _ = BaseModel.from_config(str(config), data=TheData)
    model.set_sigma(0)
    model.checkpoint_every_batches = 0
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.reconstruction_placement = "eager"
    return model


def reset(model, *, packed, final, batch):
    model.flush_word_buffers()
    if packed:
        model.dispatch_packed_soft_reset([final] * batch)
    model.dispatch_per_row_reset([final] * batch)
    model.dispatch_soft_reset()
    model.post_tick_compact()


def measure_layout(model, rows, *, packed):
    """Read roots at the inverse boundary, before output clears transient STM."""
    from What import What

    from reading_fixtures import capture_readings
    records = {}
    reconstruct = model._reconstruct_sentences
    captured = {}

    def capture(S, reference, roots, depths=None, end=None, end_depth=None, sentence_slot=None, **kwargs):
        # Reconstruction is scored once per sentence. A later sentence can
        # be absent from a shorter batch row, so it cannot overwrite that
        # row's last live sentence with its own empty end-state slot.
        captured[int(sentence_slot)] = dict(
            roots=roots.detach().clone(), end=end.detach().clone(),
            depth=depths.detach().clone(), end_depth=end_depth.detach().clone())
        return reconstruct(S, reference, roots, depths, end, end_depth, sentence_slot, **kwargs)

    model.__dict__.pop("_reconstruct_compiled", None)
    model._reconstruct_sentences = capture
    steps = [rows] if packed else [
        [row[i] + (" " if i < len(row) - 1 else "") if i < len(row) else ""
         for row in rows] for i in range(max(map(len, rows)))]
    try:
        for step, samples in enumerate(steps):
            raw = (model.inputSpace.prepPackedInput(samples) if packed
                   else model.inputSpace.prepInput(samples))
            captured.clear()
            with capture_readings(model) as readings, torch.no_grad():
                model.runBatch(
                    train=False, batchNum=step, batchSize=len(rows), split="validation",
                    batch_override=(raw, torch.empty(len(rows), 0)),
                    questions=tuple(What.present(b, split="validation") for b in range(len(rows))))
            isp = model.inputSpace
            active, ids = isp._word_active_mask, isp._packed_sentence_ids
            understood = model._last_understanding
            for b, row in enumerate(rows):
                sentences = range(len(row)) if packed else ([step] if step < len(row) else [])
                for sentence in sentences:
                    slot = sentence if packed else 0
                    mask = active[b] & (ids[b] == slot)
                    last = sentence == len(row) - 1 if packed else True
                    captured_sentence = captured[slot]
                    root = (captured_sentence["end"][b] if last
                            else captured_sentence["roots"][b, slot].reshape(3, -1))
                    program = readings[slot][b]
                    records[b, sentence] = {
                        "root": root.clone(),
                        "program_root": program.end_state.detach().clone(),
                        "actions": program.actions.detach().clone(),
                        "reference": model._tensor_pushed_ideas[b, mask].detach().clone(),
                        "recovered": model._recon_ideas[b, mask].detach().clone(),
                        "byte_cost": model._recon_sentence_costs[b, slot].detach().clone(),
                        "truncated": model._recon_truncated[b].detach().clone(),
                    }
            reset(model, packed=packed, final=step == len(steps) - 1, batch=len(rows))
    finally:
        model._reconstruct_sentences = reconstruct
        model.__dict__.pop("_reconstruct_compiled", None)
    return records


def warm_vocabulary(model, rows):
    surfaces = [s for row in rows for s in row]
    with torch.no_grad():
        model.runBatch(train=False, batchSize=len(surfaces), split="validation",
                       batch_override=(model.inputSpace.prepInput(surfaces),
                                       torch.empty(len(surfaces), 0)))
    reset(model, packed=False, final=True, batch=len(surfaces))


@pytest.mark.slow
@pytest.mark.parametrize("initialization", range(3))
def test_native_packing_preserves_each_sentence_and_owned_program(tmp_path, initialization):
    # Repeated ambient initializations, never a selected or pinned seed.
    model = build_model(tmp_path)
    # Ragged packs include a row ending at the first sentence slot, repeated
    # words and a later sentence with a multi-digit unit split.
    rows = [["9 plus 1", "14 plus 1"], ["2 plus 1"]]
    try:
        model._install_unit_span_fn()
        warm_vocabulary(model, rows)
        # Each layout starts with the same SEEN history. End/reset preserves
        # priming by design; running the second arm on the first one's history
        # changes the activated candidate bank even at unchanged parameters.
        surfaces = [(space, getattr(space, '_priming_boosts', None)) for space in model.spaces]
        snapshots = [(space, None if value is None else value.clone()) for space, value in surfaces]
        packed = measure_layout(model, rows, packed=True)
        for space, value in snapshots:
            object.__setattr__(space, '_priming_boosts', None if value is None else value.clone())
        single = measure_layout(model, rows, packed=False)
        for key, actual in packed.items():
            expected = single[key]
            assert not actual["truncated"] and not expected["truncated"]
            for name in ("reference", "actions", "root", "recovered", "byte_cost"):
                torch.testing.assert_close(actual[name], expected[name], atol=ATOL, rtol=RTOL,
                                           msg=lambda msg: f"{key} {name}: {msg}")
            torch.testing.assert_close(actual["program_root"], actual["root"], atol=0, rtol=0)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_excluded_byte_candidate_preserves_value_and_gradient():
    from types import SimpleNamespace
    from Models import BasicModel

    owner = SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
    target = torch.tensor([[[97, 0, 0]]])
    expected = expected_gradient = None
    for foreign in (False, True):
        idea = torch.tensor([[1., .3]], requires_grad=True)
        bank = torch.tensor([[[1., 0.], [float("nan"), float("nan")]]])
        surfaces = torch.tensor([[[97, 0, 0], [98, 0, 0]]])
        valid = torch.tensor([[[True, False, False], [False, False, False]]])
        if not foreign:
            bank, surfaces, valid = bank[:, :1], surfaces[:, :1], valid[:, :1]
        value = BasicModel._byte_word_cost(
            owner, idea, torch.tensor(0), bank, surfaces, valid,
            target, target != 0, True)
        gradient, = torch.autograd.grad(value.sum(), (idea,))
        assert torch.isfinite(value).all() and torch.isfinite(gradient).all()
        if expected is None:
            expected, expected_gradient = value.detach(), gradient
        else:
            torch.testing.assert_close(value, expected, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(gradient, expected_gradient, atol=ATOL, rtol=RTOL)
