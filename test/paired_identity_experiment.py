"""The §13 paired stages 1–2 pass and its conditional order-one identity lesson.

Seed, lexical order, 76 presentations and learning rate are inherited from
identity_experiment. Every presentation uses the native explore/keep optimizer
path. No stage 3/4 or order-two recovery experiment is run in this pass.
"""
import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pytest
import torch

from attention_curriculum_gates import PERCEPTS, INTERRUPTION, NOVEL, present
from identity_corpus import ROOT, KINDS, PROPERTIES, VERBS
from identity_experiment import (make_model, save, reset, present_documents,
                                 admission_ablation, component_signal)
from identity_measurement import load_split, summarize


def singleton_report(model):
    path = ROOT/'test/fixtures/2026-10-15-item6-corrected-warmup/run_warmup.py'
    spec = importlib.util.spec_from_file_location('singleton_measurement', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.singletons(model)


def columns(component):
    from IdentityRows import witnesses
    store = component.owner._closed_clause_store()
    return [dict(column=identity, witnesses=[store.occurrence_of(row) for row in witnesses(component.owner, identity)])
            for identity in component.individual_ids]


@contextmanager
def kept_programs(model, records, current):
    observe = model._sentence_observation
    def observed(state, sid, active, **kwargs):
        result = observe(state, sid, active, **kwargs)
        if kwargs.get('admit'):
            chart = model._concept_owner().components._ready()
            start, end = chart.meaning_start, chart.content_width
            positions = model._derivation_program(int(sid))[0]
            for b, entry in enumerate(result['entries']):
                if entry is None:
                    continue
                steps = []
                for index, (kind, operation, leaf) in enumerate(entry.actions.tolist()):
                    if kind < 0:
                        break
                    rules = (model.languageSpace._compose_binary_rules if kind == 1 else
                             model.languageSpace._compose_unary_rules if kind == 2 else ())
                    frame = entry.operation_values[index, :, start:end].detach()
                    value = (model._tensor_pushed_ideas[b, positions[b, leaf], start:end].detach()
                             if kind == 0 else None)
                    steps.append(dict(step=index, kind=kind, operation=operation,
                        method=rules[operation].method_name if rules else None,
                        leaf_position=int(positions[b, leaf]) if kind == 0 else None,
                        leaf_meaning_norm=float(value.norm()) if value is not None else None,
                        meaning_norms=frame.norm(dim=-1).tolist() if kind else None))
                clause = result['clauses'][b]
                value = clause.point if clause.point is not None else result['meanings'][b].roles.flatten()
                from Occurrence import source_at
                document, source_sentence = source_at(model, b, int(sid))
                records.append(dict(batch=current.get('batch'), phase=current['phase'], sentence=int(sid), row=b,
                    document=document, source_sentence=source_sentence,
                    word=current['fields'][b] if current.get('fields') else None,
                    closed_meaning_norm=float(value[start:end].detach().norm()),
                    program=[step['method'] for step in steps if step['kind']], steps=steps))
        return result
    model._sentence_observation = observed
    try:
        yield
    finally:
        model._sentence_observation = observe


def identity(model, folder, current, records):
    folder.mkdir()
    train, labels = load_split('train')
    # Preserve the accepted revision-3 branch's exact document order.
    lessons = ([d for d in train if labels[d['id']]['family'] == 'prediction_vocabulary']
               + [d for d in train if labels[d['id']]['family'] == 'pronoun']
               + [d for d in train if labels[d['id']]['family'] in ('determiner', 'same_kind')])
    assert len(lessons) == 136
    component = model._concept_owner().components
    before = set(component.individual_ids)
    current.clear(); current.update(phase='identity-training')
    optimizer = model.getOptimizer(lr=.003)
    with admission_ablation(model, False) as admission:
        present_documents(model, lessons, optimizer=optimizer, log=folder)
        trained_columns = columns(component)
        model._concept_owner()._online_learning_frozen = True
        current['phase'] = 'identity-evaluation'
        docs, labels = load_split('eval')
        reverse, reverse_labels = load_split('reversed_eval')
        labels.update(reverse_labels)
        results, observations, _ = present_documents(model, docs+reverse, labels=labels, log=folder)
    report = dict(branch='identity-balanced-on', gate='on', training_documents=len(lessons),
        evaluation_documents=len(results), columns=trained_columns,
        columns_minted=[col for col in trained_columns if col['column'] not in before],
        admission=dict(admission), summary=summarize(results), results=results, completed=True)
    save(folder/'results.json', report)
    save(folder/'observations.json', observations)
    save(folder/'kept-programs.json', [r for r in records if r['phase'].startswith('identity')])
    model.save_weights(str(folder/'final.ckpt'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    schedule = [('first', PERCEPTS), ('repeat', PERCEPTS), ('interruption', INTERRUPTION),
                ('after_gap', PERCEPTS), ('known', PERCEPTS), ('novel_first', NOVEL), ('novel_second', NOVEL)]
    atoms = list((*KINDS, *PROPERTIES, *VERBS))
    schedule += [(f'lexical:{repeat}:{start}', atoms[start:start+4])
                 for repeat in range(4) for start in range(0, len(atoms), 4)]
    reports, records, current = [], [], dict(phase='warmup')
    updates = 0
    started = time.monotonic()
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        optimizer = model.getOptimizer(lr=.003)
        step = optimizer.step
        def updated(*values, **kwargs):
            nonlocal updates
            result = step(*values, **kwargs)
            updates += 1
            return result
        optimizer.step = updated
        try:
            with kept_programs(model, records, current):
                for batch, (name, fields) in enumerate(schedule):
                    current.update(batch=batch, fields=fields)
                    result = present(model, dict(fields=fields), train=True, optimizer=optimizer)
                    assert len(result['trials']) == 2, 'paired training did not expose both trials'
                    reports.append(dict(name=name, fields=fields, training=True, **result))
                    save(args.out/'results.json', reports)
                    save(args.out/'singleton-presentations.json', records)
                    print(dict(batch=batch+1, total=19, name=name, optimizer_steps=updates), flush=True)
                assert sum(len(fields) for _, fields in schedule) == 76 and updates > 0
                with torch.no_grad():
                    report = singleton_report(model)
                latest = {record['word']:record for record in records}
                assert len(report['singleton_rows']) == len(latest) == 24
                for row in report['singleton_rows']:
                    row['kept_program'] = latest[row['word']]['program']
                    row['last_presentation_batch'] = latest[row['word']]['batch']
                count = sum(row['cosine'] is not None and row['cosine'] >= .9
                            for row in report['singleton_rows'])
                report.update(checkpoint='paired stages 1–2', seed=20261010, presentations=76,
                    optimizer_steps=updates, learning_rate=.003, completed=True,
                    elapsed_seconds=time.monotonic()-started, columns=columns(model._concept_owner().components),
                    gate=dict(carrying=count, required=20, total=24, threshold=.9, passed=count>=20))
                save(args.out/'singletons.json', report)
                save(args.out/'coordinate-signal.json', component_signal(model))
                reset(model)
                model.save_weights(str(args.out/'warmup.ckpt'))
                print(report['gate'], flush=True)
                if count >= 20:
                    identity(model, args.out/'identity-balanced-on', current, records)
                else:
                    zero = next((row['word'] for row in report['singleton_rows'] if row['closed_meaning_norm'] == 0), None)
                    if zero is None:
                        save(args.out/'zero-singleton-trace.json', dict(reason='no zero singleton; cosine gate failed'))
                    else:
                        save(args.out/'zero-singleton-trace.json', latest[zero])
                    save(args.out/'identity-not-run.json', dict(reason='singleton gate failed', gate=report['gate']))
        except BaseException:
            import traceback
            save(args.out/'failure.json', dict(error=traceback.format_exc(), optimizer_steps=updates))
            save(args.out/'kept-programs-partial.json', records)
            model.save_weights(str(args.out/'failed-state.ckpt'))
            raise
        finally:
            model.End()


if __name__ == '__main__':
    main()
