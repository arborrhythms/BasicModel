"""One ordinary batch validates audit wiring; it is not a gate training."""
import json
import os
from pathlib import Path


def test_margin_and_ownership_observe_the_same_actual_steps():
    from test_mm_xor import _fresh_model
    import review13_gate_observer
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    optimizer = model.getOptimizer(lr=.01)
    raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
    batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
    model.runBatch(train=True, optimizer=optimizer, batchSize=4, split='train', batch_override=batch)
    review13_gate_observer.save_ownership(model)
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
    retired = model._sentence_cost_registry._terms['reconstruction.antipode']
    assert retired['trained'] is False
    assert float(model._sentence_cost_registry._value(retired).abs().sum()) == 0.
    assert model._recon_antipode_cost is None
    ownership = json.loads((folder/'ownership.json').read_text())
    native = [row for row in ownership['parameters']
              if row['parameter'].startswith('perceptualSpace.') and row['writers']]
    assert native and all(row['writers'] == ['reconstruction'] for row in native)
    assert all(row['binary_rule_names'] == ['conjunction', 'disjunction'] for row in first)
    assert all(row['binary_rule_ids'] == [1, 2] for row in first)
    stability = json.loads((folder/'derivation-stability.json').read_text())
    assert len(stability) == 4
    assert all(step['rule_name'] in ('conjunction', 'disjunction', 'not')
               for row in stability for step in row['modal_derivation'])
    with review13_gate_observer.capture_final_derivations() as final:
        model.runBatch(train=False, optimizer=None, batchSize=4, split='test', batch_override=batch)
    assert len(final) == 4
    assert all(row['sequence'] for row in final.values())
    assert all(step['rule_name'] in ('conjunction', 'disjunction', 'not')
               for row in final.values() for step in row['sequence'])
    (folder/'final-greedy-compose.json').write_text(json.dumps(list(final.values()), indent=2)+'\n')
    observer = review13_gate_observer.OBSERVER.P
    observer.geometry(model, 'end')
    for phase in ('start', 'end'):
        geometry = json.loads((folder/f'geometry-{phase}.json').read_text())
        supports = [row for book in geometry['dictionary']
                    for row in book['word_perceptual_support']]
        assert {row['word'] for row in supports} == {'hello', 'world', 'loving', 'there'}
        assert all(row['dimension'] == 6 and 'nonzero_fraction' in row
                   and 'minimum_absolute_value' in row for row in supports)
