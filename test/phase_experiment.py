"""One retained first draw of the six case placements, with no parse forcing."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'),
    str(ROOT/'doc/benchmarks/2026-10-23-item3-5-step2c/development'),
    str(ROOT/'doc/benchmarks/2026-10-21-item3-5-reference-reconstruction')]
import numpy as np
import pytest
import torch
from arrival_capture import plain
from bounded_tests import source_snapshot
from constituent_corpus import CASE_VARIANTS, case_corpus
from constituent_budget import paired_updates_only
from identity_experiment import make_model, reset


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2)+'\n')


def rng():
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())


def restore_rng(value):
    torch.set_rng_state(value['torch'])
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--variant', choices=CASE_VARIANTS, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--initial-rng', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if not args.smoke and subprocess.check_output(['git', 'diff', source, '--name-only', '--', 'bin', 'test', 'data'], text=True).strip():
        raise RuntimeError('the corpus gate requires frozen implementation and fixtures')
    audit = source_snapshot(ROOT)
    if args.initial_rng is not None:
        if args.initial_rng.exists():
            restore_rng(torch.load(args.initial_rng, weights_only=False))
        else:
            args.initial_rng.parent.mkdir(parents=True, exist_ok=True)
            with args.initial_rng.open('xb') as stream:
                torch.save(rng(), stream)
    torch.set_num_threads(1)
    torch.save(rng(), args.out/'initial-rng.pt')
    protocol = dict(source=source, variant=args.variant, development=args.smoke,
        epochs=1 if args.smoke else 3, batch_size=2, learning_rate=.001,
        menu='complete production menu', answer_supervision='questions only',
        grammar_lessons=False, reference_orders=False, segmentation_hints=False,
        premise_answer_targets=False, seed_policy='one captured first draw shared across placements',
        learning_failure='retain and carry to item 0; no retry or budget extension',
        gates=dict(exact_answers=1., surface_reverse_R=0., selections='one per requested selection'))
    save(args.out/'protocol.json', protocol)
    save(args.out/'source-audit.json', audit)
    rows = case_corpus(args.variant)
    save(args.out/'corpus.json', rows)
    train = [row for row in rows if not row['heldout']]
    evaluation = [row for row in rows if row['heldout']]
    if args.smoke:
        train, evaluation = train[:2], evaluation[:2]
    progress, results, keeps = [], [], []
    started = time.monotonic()
    from Models import BasicModel
    from What import What
    from AttentionLesson import prompt_need
    from identity_keep_probe import clause_graph
    with pytest.MonkeyPatch.context() as patch:
        model = make_model(args.out, patch)
        optimizer = model.getOptimizer(lr=protocol['learning_rate'])
        def snapshot():
            return dict(state_dict=model.state_dict(), vocab_extras=model._collect_vocab_extras(),
                structural_extras=model._collect_structural_extras(), rng=rng(), optimizer=optimizer.state_dict())
        torch.save(snapshot(), args.out/'initial.pt')
        original = BasicModel._commit_sentence
        def keep(current, state, sid, active, observations, predictions, wins):
            credit = current._last_sentence_credit
            trials = [dict(clauses=[clause_graph(c) for c in item['clauses']],
                programs=[None if entry is None else dict(actions=entry.actions, word_ids=entry.word_ids,
                    case_values=entry.case_values, case_selected=entry.case_selected)
                    for entry in item['entries']], plans=item.get('reference_plans')) for item in observations]
            result = original(current, state, sid, active, observations, predictions, wins)
            record = plain(dict(batch=len(progress), sentence=int(sid), active=active, wins=wins,
                credit=credit, trials=trials, pool=current._concept_owner().mtm.state()))
            with gzip.open(args.out/'keep.jsonl.gz', 'at') as stream:
                stream.write(json.dumps(record)+'\n')
            keeps.append(record)
            return result
        patch.setattr(BasicModel, '_commit_sentence', keep)
        try:
            with paired_updates_only(model, optimizer) as updates:
                def present(batch, training, epoch):
                    split = 'train' if training else 'validation'
                    data = model.inputSpace.data
                    data.grammar_lessons = {}
                    texts = [text for row in batch for text in (*row['sentences'], row['prompt'])]
                    data.train_input = texts if training else []
                    data.train_output = [torch.zeros(1) for _ in texts] if training else []
                    data.source_addresses = {split: [dict(document=row['id'], sentence=sid,
                        document_key=row['id'], sentence_index=sid, split=split)
                        for row in batch for sid in range(3)]}
                    torch.save(snapshot(), args.out/'last-batch-start.pt')
                    torch.save(rng(), args.out/f'batch-{len(progress):03d}-rng.pt')
                    keeps.clear()
                    data.has_supervised_outputs = False
                    data.text_answers = {}
                    with torch.set_grad_enabled(training):
                        inputs = model.inputSpace.prepPackedInput([row['sentences'] for row in batch])
                        model.runBatch(train=training, optimizer=optimizer if training else None,
                            batchNum=2*len(progress), batchSize=len(batch), split=split,
                            batch_override=(inputs, torch.empty(len(batch), 0)),
                            source_rows=[[3*b, 3*b+1] for b in range(len(batch))],
                            questions=tuple(What.present(3*b+1, split=split) for b in range(len(batch))))
                    premise_keeps = tuple(keeps)
                    prompts = [row['prompt'] for row in batch]
                    data.has_supervised_outputs = training
                    data.text_answers = {'train': [answer for row in batch for answer in
                        (None, None, row['answer'])]} if training else {}
                    factory = What.supervised if training else What.inference
                    questions = tuple(factory(3*b+2, split=split, prompt=prompt) for b,prompt in enumerate(prompts))
                    with torch.set_grad_enabled(training), prompt_need(model, prompts):
                        inputs = model.inputSpace.prepPackedInput([[prompt] for prompt in prompts])
                        model.runBatch(train=training, optimizer=optimizer if training else None,
                            batchNum=2*len(progress)+1, batchSize=len(batch), split=split,
                            batch_override=(inputs, torch.zeros(len(batch), 1) if training else torch.empty(len(batch), 0)),
                            source_rows=[[3*b+2] for b in range(len(batch))], questions=questions)
                    answers = list(getattr(getattr(model, '_last_answer_construction', None), 'texts', ()) or ())
                    if not training:
                        for b,row in enumerate(batch):
                            observed = []
                            for record in premise_keeps:
                                if not record['active'][b]:
                                    continue
                                trial = int(record['wins'][b])
                                graph = record['trials'][trial]['clauses'][b]
                                selected = sum(term['operation'] in ('compound', 'compound_unary')
                                               for term in graph['compositions'])
                                observed.append(dict(sentence=record['sentence'], selections=selected,
                                    reconstruction=record['credit']['components'][b][trial][0]))
                            answer = answers[b] if b < len(answers) else None
                            selection_exact = bool(observed) and observed[-1]['selections'] == row['labels']['selections']
                            results.append(dict(id=row['id'], expected=row['answer'], answer=answer,
                                answer_exact=answer == row['answer'], observations=observed,
                                selection_exact=selection_exact,
                                reverse_exact=bool(observed) and all(item['reconstruction'] == 0. for item in observed)))
                    progress.append(dict(batch=len(progress), training=training, epoch=epoch,
                        ids=[row['id'] for row in batch], updates=dict(updates), elapsed=time.monotonic()-started))
                    save(args.out/'progress.json', progress)
                    print(json.dumps(progress[-1]), flush=True)
                    reset(model, len(batch))
                for epoch in range(protocol['epochs']):
                    model.begin_presentation()
                    for offset in range(0, len(train), protocol['batch_size']):
                        present(train[offset:offset+protocol['batch_size']], True, epoch)
                torch.save(snapshot(), args.out/'trained.pt')
                model.languageSpace.parts_of_speech.frozen = True
                model._concept_owner()._online_learning_frozen = True
                for offset in range(0, len(evaluation), protocol['batch_size']):
                    present(evaluation[offset:offset+protocol['batch_size']], False, None)
                report = dict(source=source, completed=True, development=args.smoke,
                    variant=args.variant, results=results, updates=dict(updates), total=len(results),
                    exact_answers=sum(row['answer_exact'] for row in results),
                    exact_selections=sum(row['selection_exact'] for row in results),
                    exact_reverses=sum(row['reverse_exact'] for row in results),
                    passed=all(row['answer_exact'] and row['selection_exact'] and row['reverse_exact'] for row in results),
                    elapsed_seconds=time.monotonic()-started)
                save(args.out/'results.json', report)
                print(json.dumps({key:value for key,value in report.items() if key != 'results'}), flush=True)
        except BaseException:
            torch.save(snapshot(), args.out/'failure.pt')
            save(args.out/'failure.json', dict(error=traceback.format_exc(), progress=progress))
            raise
        finally:
            save(args.out/'end.json', dict(source=source, unchanged=source_snapshot(ROOT) == audit,
                finished=datetime.now(timezone.utc).isoformat()))
            model.End()


if __name__ == '__main__':
    main()
