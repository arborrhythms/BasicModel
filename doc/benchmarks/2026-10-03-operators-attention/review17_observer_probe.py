"""One ordinary batch validates audit wiring; it is not a gate training."""
import json
import os
from pathlib import Path


def test_margin_and_ownership_observe_the_same_actual_steps():
    from test_mm_xor import _fresh_model
    import review17_gate_observer
    model, _, data = _fresh_model('data/XOR_grammar.xml')
    optimizer = model.getOptimizer(lr=.01)
    owners=model.objective_parameter_groups(optimizer)
    assert 'compose_preference' not in owners
    preference=owners['reconstruction']
    assert any(p is model.languageSpace.decomposition_chooser.weight for p in preference)
    operation=model.languageSpace._tree_layer(2)
    assert all(any(p is anchor for p in preference) for anchor in
               (operation.stop_anchor,operation.reduce_anchor,operation.apply_anchor))
    raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
    batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
    from review17_run_audit import observe_run, json_value
    with observe_run() as run_audit:
        model.runEpoch(optimizer=optimizer,batchSize=4,split='train')
        model.runBatch(train=False,optimizer=None,batchSize=4,split='test',batch_override=batch)
    decomposition=run_audit['decomposition_chooser']
    assert decomposition['start']['weights'] == [1., 0., 0., 0., 0.]
    assert decomposition['start']['undos'] > 0
    assert decomposition['start']['true_pair_in_shortlist_rate'] == 1.
    assert decomposition['end']['undos'] > 0
    steps=run_audit['compose_score_function_steps']
    assert len(steps)==4
    assert run_audit['chooser_logit_ranges']['1']['calls']>0
    assert all(row.get('gradient_max_error',0.)<2e-6 for row in steps)
    assert all(row.get('finite_difference',{}).get('error',0.)<2e-6 for row in steps)
    assert len(run_audit['reader_weights']) == 1
    assert len(run_audit['before_learning_readback']['texts']) == 4
    assert all(row['nonzero']==0 for row in run_audit['sentence_gradients'].values())
    assert all(phase in run_audit for phase in ('start','end'))
    assert all(phase in run_audit['room'] for phase in ('start','end'))
    (Path(os.environ['OWNERSHIP_OBSERVER_OUTPUT'])/'run-audit.json').write_text(json.dumps(run_audit,default=json_value,indent=2))
    review17_gate_observer.save_ownership(model)
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
    assert 'reconstruction.antipode' not in model._sentence_cost_registry._terms
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
    with review17_gate_observer.capture_final_derivations() as final:
        model.runBatch(train=False, optimizer=None, batchSize=4, split='test', batch_override=batch)
    assert len(final) == 4
    assert all(row['sequence'] for row in final.values())
    assert all(step['rule_name'] in ('conjunction', 'disjunction', 'not')
               for row in final.values() for step in row['sequence'])
    (folder/'final-greedy-compose.json').write_text(json.dumps(list(final.values()), indent=2)+'\n')
    observer = review17_gate_observer.OBSERVER.P
    observer.geometry(model, 'end')
    for phase in ('start', 'end'):
        geometry = json.loads((folder/f'geometry-{phase}.json').read_text())
        supports = [row for book in geometry['dictionary']
                    for row in book['word_perceptual_support']]
        assert {row['word'] for row in supports} == {'hello', 'world', 'loving', 'there'}
        assert all(row['dimension'] == 6 and 'nonzero_fraction' in row
                   and 'minimum_absolute_value' in row for row in supports)
