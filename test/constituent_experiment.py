"""Fixed stage 4e pass; stops at the constituent gate, never runs identity."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time
import traceback
import warnings

import numpy as np
import pytest
import torch

from constituent_corpus import DEST, load
from constituent_measurement import observe, grade, summarize
from identity_experiment import make_model, restore_warmup, reset, save

SEED, EPOCHS, RATE = 20261010, 2, .003


def lexical_orders(model, forms):
    owner = model._concept_owner()
    return {form: [dict(identity=int(cid), order=int(owner._concept_source_order(cid)))
                  for cid in owner.word_concepts(form.encode())]
            for form in sorted(forms)}


def run(args):
    from Models import BaseModel
    from What import What
    restore = BaseModel.load_weights
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train, _ = load('train')
    evaluation, labels = load('eval')
    lessons = json.loads((DEST/'train.lessons.json').read_text())
    assert len(train) == len(lessons) == 72 and len(evaluation) == 40
    forms = {word for row in train+evaluation for word in row['text'].split() if word != '.'}
    pilot = getattr(args, 'pilot', False)
    protocol = dict(seed=SEED, epochs=1 if pilot else EPOCHS,
        training_presentations=16 if pilot else len(train)*EPOCHS,
        evaluation_presentations=0 if pilot else len(evaluation), batch_size=4, learning_rate=RATE,
        warmup=str(args.warmup), candidate=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        gates=dict(brackets=.9, occurrences=.95, duplicate_writes=0,
                   reductions='one order-1 occurrence at every lowered NP; report description stamp',
                   unlowered_occurrences=0),
        attention='student-selected; no forced source-reading lesson', pilot=pilot,
        pilot_gate='both fixed-state raw mean wait and fold CEs decrease by more than 1e-5 nats',
        pilot_state='disposable; full run restarts the original warm-up and seed')
    save(args.out/'protocol.json', protocol)
    started = time.monotonic()
    records, progress, results, steps = [], [], [], [0]
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = make_model(args.out, patch)
        try:
            drift = restore_warmup(model, args.warmup, restore)
            save(args.out/'start.json', dict(loader_drift=drift, lexical_orders=lexical_orders(model, forms)))
            from GrammarLessons import _annotation_tree
            for lesson in lessons:
                _annotation_tree(model.languageSpace, lesson['tree'], len(lesson['text'].split()))
            data = model.inputSpace.data
            optimizer = model.getOptimizer(lr=RATE)
            native_step = optimizer.step
            def step(*values, **kwargs):
                result = native_step(*values, **kwargs)
                steps[0] += 1
                return result
            optimizer.step = step
            if pilot:
                import GrammarLessons
                from constituent_pilot import PilotProbe
                probe = PilotProbe(model, args.out)
                native_compose = GrammarLessons.compose_loss
                def compose_probe(language, programs, annotations, **kwargs):
                    probe.capture(language, programs, annotations)
                    return native_compose(language, programs, annotations, **kwargs)
                patch.setattr(GrammarLessons, 'compose_loss', compose_probe)
            rng = np.random.default_rng(SEED)
            lesson_audits = []
            native_lesson = model._grammar_lesson_objectives
            def lesson_audit(*values, **kwargs):
                result = native_lesson(*values, **kwargs)
                if result:
                    lesson_audits.append(dict(batch=len(progress), source=kwargs['source_rows'],
                        terms=model._grammar_lesson_errors['compose'].breakdown()))
                return result
            model._grammar_lesson_objectives = lesson_audit

            def present(texts, indices, *, training, epoch):
                split = 'train' if training else 'validation'
                data.has_supervised_outputs = False
                data.grammar_lessons = {'train': lessons} if training else {}
                data.source_addresses = {split: [dict(document=row['id'], sentence=0,
                    document_key=row['id'], sentence_index=0, split=split) for row in texts]}
                batch = [texts[index] for index in indices]
                with observe(model) as captured, torch.set_grad_enabled(training):
                    inputs = model.inputSpace.prepPackedInput([[row['text']] for row in batch])
                    model.runBatch(train=training, optimizer=optimizer if training else None,
                        batchNum=len(progress), batchSize=len(batch), split=split,
                        batch_override=(inputs, torch.empty(len(batch), 0)),
                        source_rows=[[int(index)] for index in indices],
                        questions=tuple(What.present(b, split=split) for b in range(len(batch))))
                for b, row in enumerate(batch):
                    observation = captured.get((0, b))
                    records.append(dict(id=row['id'], training=training, epoch=epoch, observed=observation))
                    if not training:
                        results.append(grade(labels[row['id']], observation))
                raw = model._last_sentence_field
                if training and len(raw['trials']) != 2:
                    raise RuntimeError('stage 4e did not run both native trials')
                lesson_reports = getattr(model, '_reading_lesson_reports', ())
                report = dict(batch=len(progress)+1, training=training, epoch=epoch,
                    ids=[row['id'] for row in batch], optimizer_steps=steps[0],
                    lesson_costs=[float(value) for value in lesson_reports] if training else [],
                    elapsed_seconds=time.monotonic()-started)
                progress.append(report)
                save(args.out/'progress.json', progress)
                save(args.out/'kept-programs.json', records)
                save(args.out/'lesson-audits.json', lesson_audits)
                print(json.dumps(report), flush=True)
                reset(model, len(batch))

            for epoch in range(1 if pilot else EPOCHS):
                order = rng.permutation(len(train)).tolist()
                if pilot:
                    order = order[:16]
                for offset in range(0, len(order), 4):
                    present(train, order[offset:offset+4], training=True, epoch=epoch)
            if pilot:
                report = dict(completed=True, protocol=protocol, optimizer_steps=steps[0],
                    elapsed_seconds=time.monotonic()-started, **probe.finish())
                save(args.out/'results.json', report)
                print(json.dumps(report), flush=True)
                return
            model.save_weights(str(args.out/'trained.ckpt'))
            model._concept_owner()._online_learning_frozen = True
            for offset in range(0, len(evaluation), 4):
                present(evaluation, list(range(offset, offset+4)), training=False, epoch=None)
            report = dict(completed=True, protocol=protocol, optimizer_steps=steps[0],
                elapsed_seconds=time.monotonic()-started, **summarize(results), results=results,
                lexical_orders=lexical_orders(model, forms))
            save(args.out/'results.json', report)
            print(json.dumps(dict(completed=True, gates=report['gates'], passed=report['passed'])), flush=True)
        except BaseException:
            save(args.out/'failure.json', dict(error=traceback.format_exc(),
                completed_batches=len(progress), optimizer_steps=steps[0],
                completed_training_presentations=sum(row['training'] for row in records),
                completed_evaluation_presentations=len(results), elapsed_seconds=time.monotonic()-started))
            raise
        finally:
            model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--warmup', type=Path,
        default=Path('output/item6-orders-paired-accepted-code-20261015/warmup.ckpt'))
    parser.add_argument('--pilot', action='store_true', help='16 presentations; fixed teacher-state CE gate, no held-out run')
    run(parser.parse_args())
