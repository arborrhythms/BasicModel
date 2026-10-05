"""One natural seed-613 batch on frozen source; observation, not a gate.

This keeps the original chooser test's seed, text, optimizer and model setup,
without its controlled tie/non-tie cost fixtures. No cost is overwritten.
"""
import json
import os
from pathlib import Path
import torch


def test_ordinary_batch_logit_gradient(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from reading_fixtures import use_eager_reading
    from review17_score_probe import observe_score_function
    from review17_run_audit import json_value
    from bounded_tests import source_snapshot
    use_eager_reading(monkeypatch)
    torch.manual_seed(613)
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets="16", input_width=16)
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.conceptualSpace.intra_loss_weight = 0.0
    model.inter_loss_weight = 0.0
    optimizer = model.getOptimizer(lr=1e-3)
    words = ["a b c d e", "f g h i j"]
    inputs = model.inputSpace.prepInput(words)
    data = model.inputSpace.data
    supervised_before = data.has_supervised_outputs
    data.has_supervised_outputs = False
    report = dict(kind='mechanism check, not a gate', batch=words,
                  seed=613, seed_source='unchanged original chooser fixture',
                  cost_override=False, training_batches=1)
    root = Path(__file__).resolve().parents[3]
    source = source_snapshot(root)
    assert source == json.loads((Path(__file__).parent/'review17-source/source.json').read_text())
    try:
        with observe_score_function(report):
            model.runBatch(train=True, batchNum=0, batchSize=2, split='train', optimizer=optimizer,
                           batch_override=(inputs, torch.empty(2, 0)))
        report['ownership'] = model.ownership_gradient_diagnostics(optimizer)
        assert report['ownership']['conflicts'] == 0
        rows = report['compose_score_function_steps']
        assert len(rows) == 2 and all(row['departed'] for row in rows)
        assert all(row['gradient_max_error'] < 2e-6 for row in rows)
        assert all(row['finite_difference']['error'] < 2e-6 for row in rows)
        assert source == source_snapshot(root)
        report['source_matched'] = True
    finally:
        destination = Path(os.environ['REVIEW17_ORDINARY_OUTPUT'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x') as f: json.dump(report, f, indent=2, default=json_value)
        data.has_supervised_outputs = supervised_before
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
