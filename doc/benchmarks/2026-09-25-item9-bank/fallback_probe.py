"""Measure a lost-ownership inverse on a chosen source tree, without a seed."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--expect", choices=("null", "error"), required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.root / name) for name in ("bin", "test")]

import torch
from bounded_tests import source_snapshot
from test_packed_reconstruction_parity import build_model
from test_reverse_traversal import _stage_packed

with tempfile.TemporaryDirectory() as temporary:
    model = build_model(Path(temporary))
    report = dict(source=source_snapshot(args.root))
    try:
        model._install_unit_span_fn()
        _stage_packed(model, [["quorp flarn", "wug blim"]])
        with torch.no_grad():
            model._publish_compiled_sentence_state(model._forward_with_compiled_sentence_state(None))
        isp = model.inputSpace
        report["populated_candidates"] = int((isp._ar_concept_lookup_rows >= 0).sum())
        report["admission"] = model._concept_owner().concept_admission_stats()
        # The inventory is populated; only staging provenance is lost.
        isp._ar_concept_lookup_sentence_ids = None
        end = model._tensor_final_end_slots.detach().clone().requires_grad_()
        try:
            result = model._reconstruct_sentences(
                model._stm_single_S.detach(), model._tensor_pushed_ideas.detach(),
                model._tensor_sentence_roots_live.detach(), model._tensor_sentence_roots_depth,
                end, model._tensor_final_end_depth)
        except RuntimeError as exc:
            report.update(outcome="error", error=str(exc))
            assert args.expect == "error" and "sentence ownership" in str(exc)
        else:
            gradient, = torch.autograd.grad(result[2].sum(), (end,))
            report.update(outcome="null", byte_cost=result[2].detach().tolist(),
                          end_gradient_max=float(gradient.abs().max()),
                          truncated=result[3].tolist())
            assert args.expect == "null"
            torch.testing.assert_close(result[2], torch.full_like(result[2], 256.).log())
            assert not gradient.any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source"}, indent=2))
