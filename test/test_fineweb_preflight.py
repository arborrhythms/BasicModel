"""Fast launch gates for corpus-scale MM_20M FineWeb training."""

import math
import os
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT / "data" / "MM_20M_fineweb.xml"


def test_production_config_is_fineweb_and_recoverable():
    import xml.etree.ElementTree as ET

    root = ET.parse(CONFIG).getroot()
    assert root.findtext("./architecture/data/dataset") == "text"
    assert root.findtext("./architecture/data/shardDir") == "data/fineweb"
    assert root.findtext("./PartSpace/nOutput") == "64"
    assert root.findtext("./ConceptualSpace/stmCapacity") == "64"
    assert root.findtext("./architecture/training/autosave") == "true"
    assert int(root.findtext(
        "./architecture/training/checkpointEveryBatches")) > 0
    assert root.findtext("./architecture/weightsPath")
    assert root.findtext("./architecture/training/seed") is not None


def test_make_train_defaults_to_production_config():
    makefile = (PROJECT / "Makefile").read_text()
    assert "MODEL ?= data/MM_20M_fineweb.xml" in makefile


def test_train_cli_defaults_to_production_config():
    import train

    assert train.parse_args([]).model == "data/MM_20M_fineweb.xml"


def test_local_cli_forwards_every_phase_two_bound(monkeypatch):
    import train

    calls = []
    monkeypatch.setattr(train, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    args = train.parse_args([
        "--model", "data/MM_20M_fineweb.xml",
        "--data", "text", "--max-docs", "20", "--num-shards", "3",
        "--num-epochs", "2", "--batch-size", "1", "--max-tokens", "17",
        "--batches", "4", "--random-shards",
    ])
    train.train_local(args)

    _, kwargs = calls[-1]
    env = kwargs["env"]
    assert env["BASIC_DATASET"] == "text"
    assert env["BASIC_MAX_DOCS"] == "20"
    assert env["BASIC_NUM_SHARDS"] == "3"
    assert env["BASIC_NUM_EPOCHS"] == "2"
    assert env["BASIC_BATCH_SIZE"] == "1"
    assert env["BASIC_MAX_TOKENS"] == "17"
    assert env["BASIC_MAX_BATCHES"] == "4"
    assert env["BASIC_RANDOM_SHARDS"] == "1"


def test_remote_cli_preserves_safety_and_embedding_flags(monkeypatch):
    import train

    calls = []
    monkeypatch.setattr(train, "run", lambda cmd, **kw: calls.append(cmd))
    args = train.parse_args([
        "--model", "data/MM_20M_fineweb.xml", "--host", "trainer",
        "--batch-size", "2", "--batches", "9", "--max-tokens", "31",
        "--force-embeddings", "--latent-vector-size", "96",
        "--embed-lr", "0.002", "--random-shards",
    ])
    train.train_remote(args)

    ssh = next(cmd for cmd in calls
               if cmd and cmd[0] == "ssh" and "bin/train.py" in cmd[-1])
    remote = ssh[-1]
    for expected in (
        "--batch-size 2", "--batches 9", "--max-tokens 31",
        "--force-embeddings", "--latent-vector-size 96",
        "--embed-lr 0.002", "--random-shards",
    ):
        assert expected in remote, remote


def test_document_splits_do_not_leak_and_controls_reach_shards(
        tmp_path, monkeypatch):
    import data as data_mod

    docs = [f"doc{i} alpha beta gamma. doc{i} second sentence."
            for i in range(20)]
    seen = {}

    def fake_paths(path, num_shards=1, random_select=False):
        seen["random"] = random_select
        return ["fixture.parquet"]

    monkeypatch.setattr(data_mod, "get_shard_paths", fake_paths)
    monkeypatch.setattr(
        data_mod, "iter_documents",
        lambda paths, max_docs=None: iter(docs[:max_docs]))
    d = data_mod.Data()
    d.loadShards(2, 20, str(tmp_path), random_shards=True, max_tokens=3)

    assert seen["random"] is True
    split_markers = []
    for rows in (d.train_input, d.validation_input, d.test_input):
        markers = {word for row in rows for word in row.split()
                   if word.startswith("doc")}
        split_markers.append(markers)
        assert all(len(row.split()) <= 3 for row in rows)
    assert split_markers[0].isdisjoint(split_markers[1])
    assert split_markers[0].isdisjoint(split_markers[2])
    assert split_markers[1].isdisjoint(split_markers[2])
    assert d.has_supervised_outputs is False


def test_checkpoint_restores_optimizer_counters_and_rng(tmp_path):
    import Language
    import Models
    from util import init_config, init_device

    init_device("cpu")
    cfg = PROJECT / "data" / "MM_xor.xml"
    init_config(path=str(cfg), defaults_path=str(PROJECT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    Models.TheData.load("xor")
    model, _ = Models.BasicModel.from_config(str(cfg), data=Models.TheData)
    model._optimizer = model.getOptimizer(lr=1e-4)
    model.runEpoch(optimizer=model._optimizer, batchSize=2,
                   split="train", max_batches=1)
    model._train_batches_seen = 7
    assert model._epoch_batches_seen == 1
    path = tmp_path / "resume.ckpt"
    model.save_weights(str(path))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_rng = saved["training_state"]["torch_rng_state"]
    assert saved["optimizer_state"] is not None

    Language.TheGrammar._configured = False
    restored, _ = Models.BasicModel.from_config(str(cfg), data=Models.TheData)
    assert restored.load_weights(str(path), require_match=True)
    opt = restored.getOptimizer(lr=1e-4)
    assert restored._training_step_count == model._training_step_count
    assert restored._train_batches_seen == 7
    assert restored._epoch_batches_seen == 1
    assert restored._resume_batches_to_skip == 1
    assert restored._checkpoint_batch_size == 2
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert opt.state_dict()["state"]
    with pytest.raises(ValueError, match="different batch size"):
        restored.runEpoch(optimizer=opt, batchSize=1,
                          split="train", max_batches=1)
    cursor = Models.TheData.data_loader(
        split="train", num_streams=2,
        slab_bytes=int(restored.inputSpace.outputShape[0])).dataset
    total_ticks = 0
    while not cursor.all_done():
        cursor.next_tick()
        total_ticks += 1
    before = restored._training_step_count
    restored.runEpoch(optimizer=opt, batchSize=2, split="train")
    assert restored._training_step_count == before + total_ticks - 1
    assert restored._epoch_batches_seen == 0


@pytest.mark.skipif(
    os.environ.get("RUN_FINEWEB_STEP") != "1",
    reason="set RUN_FINEWEB_STEP=1 for the real N=64 optimizer-step gate")
def test_real_fineweb_config_runs_one_self_supervised_step(monkeypatch):
    import Language
    import Models
    from util import init_config, init_device

    init_device("cpu")
    init_config(path=str(CONFIG),
                defaults_path=str(PROJECT / "data" / "model.xml"))
    rows = ["The cat chases the mouse."] * 10
    corpus = {
        "train": {"text": rows[:8], "label": []},
        "validation": {"text": rows[8:9], "label": []},
        "test": {"text": rows[9:], "label": []},
    }
    Models.TheData.processLM(corpus)
    Language.TheGrammar._configured = False
    monkeypatch.setattr(Models.BaseModel, "load_weights",
                        lambda self, *a, **k: False)
    model, _ = Models.BaseModel.from_config(str(CONFIG), data=Models.TheData)
    model = model.to("cpu")
    optimizer = model.getOptimizer(lr=5e-4)
    out_loss, recon_loss, *_ = model.runEpoch(
        optimizer=optimizer, batchSize=1, split="train", max_batches=1)
    assert math.isfinite(float(out_loss))
    assert math.isfinite(float(recon_loss)) and float(recon_loss) > 0
    assert model._training_step_count == 1
    assert Models.TheData.has_supervised_outputs is False
