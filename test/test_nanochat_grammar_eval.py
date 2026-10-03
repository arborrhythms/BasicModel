"""Unit gates for the frozen NanoChat-sized grammar evaluation harness."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import torch

import eval_nanochat_grammar as gate
from Spaces import ConceptualSpace


VOCAB = [
    "amber", "birch", "cedar", "daisy", "elm", "fern", "grove",
    "hazel", "iris", "juniper", "kelp", "lilac", "maple", "nettle",
    "olive", "pine", "quince", "reed", "sage", "thyme", "umber",
    "violet", "willow", "yarrow", "zinnia",
]


def _documents():
    docs = [f"training document {i}." for i in range(9)]
    sentences = []
    for i in range(40):
        rotated = VOCAB[i % len(VOCAB):] + VOCAB[:i % len(VOCAB)]
        sentences.append(" ".join(rotated[:12]) + ".")
    docs.append(" ".join(sentences))  # doc 9 is the test split
    return docs


def test_manifest_is_deterministic_document_heldout_and_frozen():
    kwargs = dict(
        num_items=8, seed=7, heldout_split="test",
        min_prefix_words=4, max_prefix_words=8,
        num_choices=4, max_completion_bytes=96,
        shard_names=["synthetic.parquet"], max_docs=10,
    )
    first = gate.build_manifest_from_documents(_documents(), **kwargs)
    second = gate.build_manifest_from_documents(_documents(), **kwargs)
    assert first == second
    assert gate.validate_manifest(first) is first
    assert len(first["items"]) == 8
    assert {item["doc_index"] for item in first["items"]} == {9}
    assert first["metadata"]["heldout_split"] == "test"
    assert first["metadata"]["item_sha256"] == gate._manifest_digest(
        first["items"])

    for item in first["items"]:
        assert len(item["candidates"]) == 4
        assert item["candidates"][item["answer_index"]] == item["target"]
        assert Counter(gate.surface_words(item["prefix"])) == Counter(
            gate.surface_words(item["shuffled_prefix"]))
        assert item["prefix"] != item["shuffled_prefix"]


def test_manifest_round_trip_and_digest_guard(tmp_path):
    manifest = gate.build_manifest_from_documents(
        _documents(), num_items=3, seed=11, num_choices=4,
        max_completion_bytes=96, max_docs=10)
    path = tmp_path / "manifest.json"
    gate.write_manifest(path, manifest)
    assert gate.load_manifest(path) == manifest

    broken = gate.load_manifest(path)
    broken["items"][0]["target"] = "tampered"
    path.write_text(__import__("json").dumps(broken), encoding="utf-8")
    try:
        gate.load_manifest(path)
    except ValueError as exc:
        assert "answer" in str(exc) or "sha256" in str(exc)
    else:
        raise AssertionError("tampered manifest should fail validation")


def test_last_gated_intra_score_uses_final_active_word_per_row():
    trace = [
        {
            # Whole-slab sentence prelude: deliberately ignored by the
            # next-word scorer.
            "prediction": torch.zeros(2, 3, 2),
            "target": torch.ones(2, 3, 2),
            "row_gate": None,
        },
        {
            "prediction": torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
            "target": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            "row_gate": torch.tensor([[True], [False]]),
        },
        {
            "prediction": torch.tensor([[3.0, 3.0], [0.0, 0.0]]),
            "target": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            "row_gate": torch.tensor([[False], [True]]),
        },
        {
            "prediction": torch.tensor([[4.0, 4.0], [9.0, 9.0]]),
            "target": torch.tensor([[2.0, 2.0], [0.0, 0.0]]),
            "row_gate": torch.tensor([[True], [False]]),
        },
    ]
    scores, targets, predictions, counts = gate.last_gated_intra_scores(trace)
    assert torch.equal(scores, torch.tensor([4.0, 4.0]))
    assert torch.equal(targets, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
    assert torch.equal(predictions, torch.tensor([[4.0, 4.0], [0.0, 0.0]]))
    assert torch.equal(counts, torch.tensor([2, 1]))


def test_capture_hook_runs_under_no_grad_without_loss_accumulation():
    proxy = SimpleNamespace(_intra_capture=None, intra_loss_weight=0.0)
    prediction = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[3.0, 4.0]])
    gate_tensor = torch.tensor([[True]])
    with ConceptualSpace.capture_intra_predictions(proxy) as entries:
        with torch.no_grad():
            ConceptualSpace._accumulate_intra_loss(
                proxy, prediction, target, row_gate=gate_tensor)
    assert len(entries) == 1
    assert torch.equal(entries[0]["prediction"], prediction)
    assert torch.equal(entries[0]["target"], target)
    assert torch.equal(entries[0]["row_gate"], gate_tensor)
    assert proxy._intra_capture is None






def test_checkpoint_prewarm_builds_reducer_and_full_wholes_inventory():
    class FakeCodebook:
        nVectors = 8192

    codebook = FakeCodebook()
    reducer_calls = []
    to_calls = []
    model = SimpleNamespace(
        _stm_reducer=lambda: reducer_calls.append(True),
        wholeSpace=SimpleNamespace(
            nVectors=8192,
            subspace=SimpleNamespace(codebook=lambda: codebook)),
        to=lambda target: to_calls.append(target),
    )

    gate._prewarm_checkpoint_shapes(model, torch.device("cpu"))

    assert reducer_calls == [True]
    assert codebook.nVectors == 8192
    assert to_calls == [torch.device("cpu")]
