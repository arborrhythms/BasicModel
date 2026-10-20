"""One balanced order-one branch from the reviewed paired warm-up (§14).

Same seed, revision-3 documents, order, optimizer and presentation budgets.
Only native text is presented. Labels are read solely by the closing grader.
"""
import argparse
from pathlib import Path
import random
import time
import traceback
import warnings

import numpy as np
import pytest
import torch

from identity_experiment import make_model, restore_warmup, present_documents, save
from identity_measurement import load_split, summarize
from paired_identity_experiment import kept_programs, columns


def run(args):
    from Models import BaseModel
    restore = BaseModel.load_weights
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    records, current = [], dict(phase='identity-training')
    started = time.monotonic()
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.warmup, restore)
            save(args.out/'start.json', dict(warmup=str(args.warmup), seed=20261010,
                native_loader_drift=drift, new_reference_policy='neutral salience and form cues',
                old_tensors_restored=True, smoke=args.smoke))
            train, labels = load_split('train')
            lessons = ([d for d in train if labels[d['id']]['family'] == 'prediction_vocabulary']
                + [d for d in train if labels[d['id']]['family'] == 'pronoun']
                + [d for d in train if labels[d['id']]['family'] in ('determiner', 'same_kind')])
            assert len(lessons) == 136
            docs, labels = load_split('eval')
            reverse, reverse_labels = load_split('reversed_eval')
            labels.update(reverse_labels); docs += reverse
            if args.smoke:
                lessons, docs = lessons[:4], docs[:4]
            component = model._concept_owner().components
            before = set(component.individual_ids)
            optimizer = model.getOptimizer(lr=.003)
            observe = model._sentence_observation
            reference_readings = []
            def reference_observation(state, sid, active, **kwargs):
                result = observe(state, sid, active, **kwargs)
                if kwargs.get('admit'):
                    from Occurrence import source_at
                    for b, plan in enumerate(result.get('reference_plans', ())):
                        document, position = source_at(model, b, int(sid))
                        reference_readings.append(dict(document=document, sentence=position,
                            phase=current['phase'], requests=[dict(mode=item['mode'], role=item['role'],
                                leaves=item['leaves'], candidates=item['candidates'],
                                content_norm=float(item['content'].norm()),
                                predicate_norm=float(item['predicate'].norm())) for item in plan['items']]))
                    save(args.out/'reference-readings.json', reference_readings)
                return result
            model._sentence_observation = reference_observation
            with kept_programs(model, records, current):
                present_documents(model, lessons, optimizer=optimizer, log=args.out)
                trained_columns = columns(component)
                model._concept_owner()._online_learning_frozen = True
                current['phase'] = 'identity-evaluation'
                results, observations, _ = present_documents(model, docs, labels=labels, log=args.out)
            report = dict(completed=True, smoke=args.smoke, seed=20261010,
                training_documents=len(lessons), evaluation_documents=len(docs),
                training_columns=trained_columns, columns=columns(component),
                columns_minted=[c for c in columns(component) if c['column'] not in before],
                summary=summarize(results), results=results, elapsed_seconds=time.monotonic()-started,
                policy=component.reference_close.policy.detach().tolist(),
                form_regions=len(component.owner._concept_allocator.form_regions))
            save(args.out/'results.json', report)
            save(args.out/'observations.json', observations)
            save(args.out/'kept-programs.json', records)
            model.save_weights(str(args.out/'final.ckpt'))
        except BaseException:
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
            save(args.out/'kept-programs-partial.json', records)
            raise
        finally:
            model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--warmup', type=Path, default=Path('output/item6-orders-paired-accepted-code-20261015/warmup.ckpt'))
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
