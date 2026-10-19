#!/usr/bin/env python3
"""§9.1: enabled shipped defaults versus the sole declared parity configuration.

Native document/word/closing/trial paths, with optimizer updates withheld.
This measures a changed selection default; it is not another learning run.
"""
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]


def measure(output):
    import numpy as np
    import pytest
    import torch
    import util
    import ModelCandidateAttention
    from test_compiled_word_chunk import _tiny_canonical_model, _stage_fullgraph_tensor_peer
    torch.set_num_threads(1)
    declaration = json.loads((ROOT/'test/fixtures/item4-parity.json').read_text())
    corpus = [['red cat', 'blue dog'], ['cat runs', 'dog walks'],
              ['red cat runs', 'blue dog walks']]
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    report = dict(gate='9.1', declaration='test/fixtures/item4-parity.json',
                  seed=int(torch.initial_seed()), seed_selection='ambient; one paired comparison, no retries',
                  corpus=corpus, updates=0, arms={}, status='running')
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    initial = None
    for arm in ('parity', 'enabled_default'):
        torch.set_rng_state(rng[0]); random.setstate(rng[1]); np.random.set_state(rng[2])
        with tempfile.TemporaryDirectory() as folder, pytest.MonkeyPatch.context() as patch:
            patch.setattr(util, 'TheCompileBackend', 'none')
            overrides = declaration['training'] if arm == 'parity' else {}
            model = _tiny_canonical_model(Path(folder), patch, input_width=32, word_buckets='8',
                stm_capacity=4, chooser_depth=1, training_overrides={'intraLossWeight':0., **overrides})
            model._tensor_peer_while_eager = True
            head, chooser = model.symbolSpace.expectation, model._stm_reducer()
            params = {n:p.detach().clone() for n,p in model.named_parameters()}
            if initial is None:
                initial = params
            else:
                assert params.keys() == initial.keys()
                for name, value in params.items():
                    torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
            watched = {n:p.detach().clone() for n,p in model.named_parameters()
                       if 'operation_layer' in n or '_inter_predictor' in n}
            item = dict(prime=model.expectation_prime_enabled, phi=chooser.exploration_fraction,
                        gains=[model.percept_expectation_gain, model.word_expectation_gain,
                               model.expectation_gain, model.document_expectation_gain],
                        initial_parameters_matched=len(params), batches=[])
            report['arms'][arm] = item
            assert item['gains'] == [0.,0.,0.,0.]
            assert head.expectation_enabled and util.TheXMLConfig.training('wordExpectation', True)
            current = None
            def observe_step(loss):
                assert torch.isfinite(loss)
                current['would_update_losses'].append(float(loss.detach()))
            patch.setattr(model, '_sentence_train_step', observe_step)
            choose = chooser.forward
            def observe_choose(*args, **kwargs):
                prime = getattr(chooser, '_expectation_prime', None)
                if prime is not None:
                    current['compose_prime_norms'].append(float(prime.norm()))
                return choose(*args, **kwargs)
            patch.setattr(chooser, 'forward', observe_choose)
            context = ModelCandidateAttention.context
            def observe_context(owner, *args, **kwargs):
                result = context(owner, *args, **kwargs)
                if owner is model:
                    count = sum(model._candidate_geometry[0])
                    # Context: two activation, one heat, two time columns, then prime.
                    prime = result[:,5*count:7*count]
                    current['candidate_prime_norms'].append(float(prime.norm()))
                return result
            patch.setattr(ModelCandidateAttention, 'context', observe_context)
            cost = model._sentence_path_cost
            def observe_cost(*args, **kwargs):
                value = cost(*args, **kwargs)
                registry = model._sentence_cost_registry
                reconstruction = registry.total(objective='reconstruction')
                current['trials'].append(dict(R=reconstruction.detach().tolist(),
                    symbol_observations=len(value[2].get('symbol_expectation', ())),
                    roles=[None if v is None else v.detach().tolist() for v in value[2]['observed']]))
                return value
            patch.setattr(model, '_sentence_path_cost', observe_cost)
            optimizer = model.getOptimizer(lr=0.)
            data = model.inputSpace.data
            data.source_addresses['train'] = [dict(document=f'parity-document-{b}',sentence=s)
                for s in range(len(corpus)) for b in range(2)]
            try:
                for sid, samples in enumerate(corpus):
                    current = dict(sentence=sid, samples=samples, trials=[], would_update_losses=[],
                                   compose_prime_norms=[], candidate_prime_norms=[])
                    item['batches'].append(current)
                    model.train()
                    rows = [2*sid,2*sid+1]
                    model._stage_expectation_documents('train', rows, 2)
                    raw = _stage_fullgraph_tensor_peer(model, samples)
                    current['input_ids'] = model.inputSpace._ar_word_concept_ids.detach().tolist()
                    current['input_keys'] = model.inputSpace._ar_word_object_atoms.detach().tolist()
                    with model._sentence_run(training=True, optimizer=optimizer, split='train', source_rows=rows):
                        result = model._forward_with_compiled_sentence_state(raw)
                    model._publish_compiled_sentence_state(result); model._end_step()
                    audit = model._last_sentence_credit
                    current['judge'] = {key:audit[key].detach().tolist() for key in ('costs','components','wins')}
                    current['departure'] = {key:value.detach().tolist() for key,value in audit['departure'].items()
                                            if torch.is_tensor(value)}
                    current['D_predictions'] = [head.last_expectation_comparison(b) is not None for b in range(2)]
                    current['end_slots'] = model._tensor_final_end_slots.detach().tolist()
                    current['store_roles'] = [None if f is None else f.meaning.roles.detach().tolist()
                                              for f in model._sentence_fields[0]]
                    assert len(current['trials']) == 2
                    if not item['prime']:
                        assert not any(current['compose_prime_norms'])
                    model.End(); model.symbolSpace.soft_reset()
                for name, param in model.named_parameters():
                    if name in watched:
                        torch.testing.assert_close(param, watched[name], atol=0, rtol=0)
                item['chooser_and_predictor_unchanged'] = True
            finally:
                model.End(); model.symbolSpace.soft_reset()
    comparisons = []
    left, right = report['arms']['parity'], report['arms']['enabled_default']
    for a,b in zip(left['batches'], right['batches']):
        comparisons.append(dict(sentence=a['sentence'],
            input_ids_equal=a['input_ids']==b['input_ids'], input_keys_equal=a['input_keys']==b['input_keys'],
            end_slots_equal=a['end_slots']==b['end_slots'], store_roles_equal=a['store_roles']==b['store_roles'],
            judge_equal=a['judge']==b['judge'], departure_equal=a['departure']==b['departure'],
            parity_R=a['judge']['components'], enabled_R=b['judge']['components']))
    report.update(status='complete', comparisons=comparisons,
                  changed_defaults_are_not_an_exact_parity_claim=True,
                  learning_gate_unchanged=True)
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    return report

if __name__ == '__main__':
    measure(sys.argv[1])
