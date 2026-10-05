"""One ordinary batch validates audit wiring; it is not a gate training."""
import json
import os
from pathlib import Path


def test_margin_and_ownership_observe_the_same_actual_steps():
    from test_mm_xor import _fresh_model
    import review11_gate_observer
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    optimizer = model.getOptimizer(lr=.01)
    raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
    batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
    model.runBatch(train=True, optimizer=optimizer, batchSize=4, split='train', batch_override=batch)
    review11_gate_observer.save_ownership(model)
    folder = Path(os.environ['OWNERSHIP_OBSERVER_OUTPUT'])
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    first = [event for event in events if event['kind'] == 'decoder_first_logits']
    steps = [event for event in events if event['kind'] == 'decoder_margin_step']
    assert len(first) == 4
    assert {row['path'] for row in first} == {'greedy', 'explore'}
    assert len(steps) == 3
    assert sum(bool(step['walks']) for step in steps) == 2
    assert len({walk['id'] for step in steps for walk in step['walks']}) == 4
    assert json.loads((folder/'ownership.json').read_text())['conflicts'] == 0
