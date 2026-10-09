"""Full development epoch; no declared attempt and no evaluation result.

The original corpus, presentation driver, loss-side verifier and observer are
imported unchanged. This wrapper only grades document presentation, provisions
the development capacity/budget, and records timings around the real path.
"""
from collections import Counter
from dataclasses import asdict
import cProfile
import json
from pathlib import Path
import pstats
import pickle
import random
import re
import sys
import time
import traceback
import zipfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/'doc/benchmarks/2026-10-07-math-chain'
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'), str(OLD)]


def main(folder, condition, replay=None):
    import numpy as np
    import torch
    from math_chain_corpus import MathChainCorpus
    from math_train import stage
    from math_observer import Observer
    from MathChainTraining import present
    from Models import BasicModel
    from ThoughtReferences import open_slots, bindings
    from test_math_chain import build_model

    folder.mkdir(exist_ok=False)
    if replay is not None:
        with (replay/'initial-rng.pkl').open('rb') as stream:
            states = pickle.load(stream)
        torch.set_rng_state(states['torch'])
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
    # Preserve entropy, never choose a seed. Diagnostic replays can recover
    # this exact development start without replacing its recorded outcome.
    with (folder/'initial-rng.pkl').open('wb') as stream:
        pickle.dump(dict(torch=torch.get_rng_state(),python=random.getstate(),
                         numpy=np.random.get_state()),stream)
    helper_paths = [Path(__file__).resolve(), HERE/'run_development.py',
                    OLD/'math_train.py', OLD/'math_observer.py']
    import hashlib
    helpers = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in helper_paths}
    (folder/'development-helpers.json').write_text(json.dumps(helpers,indent=2)+'\n')
    with zipfile.ZipFile(folder/'development-helpers.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in helpers:
            archive.write(ROOT/name,name)
    from bounded_tests import source_snapshot
    source = source_snapshot(ROOT)
    (folder/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    with zipfile.ZipFile(folder/'development-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in source:
            archive.write(ROOT/name,name)
    torch.set_num_threads(1)
    start = time.perf_counter()
    model = None
    result = dict(kind='development', condition=condition, seed=None,
                  attention_budget=0 if condition == 'zero_attention_budget' else 32,
                  ltm_capacity=32768, completed_epochs=0, status='started')
    if replay is not None:
        result['development_replay_of'] = str(replay)
    timings, counts = Counter(), Counter()
    batch_events = {}
    declaratives = []
    exhausted_examples = dict(declarative=[], question=[])
    batch_seconds = 0.
    profiled = False
    batches = (folder/'batches.jsonl').open('w')

    def prohibited(*_args, **_kwargs):
        raise AssertionError('development learner has no explicit seed')

    try:
        with patch.object(torch, 'manual_seed', prohibited), patch.object(np.random, 'seed', prohibited):
            if replay is None:
                presentation = MathChainCorpus().presentation()
                presentation['train'] = sorted(presentation['train'],
                    key=lambda doc: -1 if doc.pair is None else doc.pair[1])
                presentation = {split: [asdict(doc) for doc in docs]
                                for split, docs in presentation.items()}
            else:
                presentation = json.loads((replay/'presentation.json').read_text())
            (folder/'presentation.json').write_text(json.dumps(presentation)+'\n')
            text = (ROOT/'data/MM_math_chain.xml').read_text()
            text = re.sub(r'<ltmCapacity>\d+</ltmCapacity>', '<ltmCapacity>32768</ltmCapacity>', text)
            text = re.sub(r'<attentionBudget>\d+</attentionBudget>',
                          f'<attentionBudget>{result["attention_budget"]}</attentionBudget>', text)
            config = folder/'development.xml'
            config.write_text(text)
            model = build_model(config)
            data = model.inputSpace.data
            stage(data, presentation, condition=condition, epoch=0, run='development')
            chooser = model._selected_thought_chooser(None)
            initial = {name: value.detach().clone() for name, value in chooser.named_parameters()}
            torch.save(initial,folder/'initial-chooser.pt')
            optimizer = model.getOptimizer(lr=.001)

            with Observer(folder) as observer:
                observer.context = dict(epoch=1, phase='development')
                ordinary_episode = BasicModel.run_selected_thought
                ordinary_batch = BasicModel.runBatch

                def episode(model, meaning, **kwargs):
                    nonlocal profiled
                    profile = cProfile.Profile() if not profiled and result['attention_budget'] else None
                    timer = time.perf_counter()
                    if profile is not None:
                        profiled = True
                        profile.enable()
                    try:
                        value = ordinary_episode(model, meaning, **kwargs)
                    finally:
                        if profile is not None:
                            profile.disable()
                            profile.dump_stats(str(folder/'thought-episode.prof'))
                            with (folder/'thought-episode-profile.txt').open('w') as stream:
                                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(70)
                    elapsed = time.perf_counter()-timer
                    operations = [record.operation for record in value.records if record.kind == 'thought']
                    checked_queries = [record.result for record in value.records
                        if record.kind == 'thought' and record.operation == 'query' and record.result is not None]
                    empty_close = (operations == ['query','conclude'] and len(checked_queries) == 1
                        and not checked_queries[0].evidence.get('frames')
                        and not checked_queries[0].evidence.get('incomplete'))
                    exhaustion = None
                    free_roles = open_slots(meaning)
                    if empty_close and free_roles and all(kind == 'referent' for kind, _ in free_roles):
                        supplied = tuple(bindings(meaning).get('_source_evidence',(0.,0.))) != (0.,0.)
                        kind = 'declarative' if supplied else 'question'
                        store = model.symbolSpace.ltm_store
                        occurrence = bindings(meaning).get('_query_occurrence')
                        index = store._index_occurrences.get(occurrence)
                        committed = None if index is None else store.meaning_of(index)
                        minted = any(bindings(child).get('_formation_reason') == 'search_exhausted'
                                     for child in value.meaning.constituents)
                        correct = ((minted and not open_slots(value.meaning)) if supplied else
                                   (bool(open_slots(value.meaning)) and committed is not None
                                    and store.KINDS[int(store.record_kind[index])] == 'question'))
                        exhaustion = dict(kind=kind,correct=correct,minted=minted,occurrence=occurrence,
                            operations=operations, open_before=open_slots(meaning),
                            open_after=open_slots(value.meaning),
                            committed_open=None if committed is None else open_slots(committed))
                        counts['exhausted_'+kind+'_observed'] += 1
                        counts['exhausted_'+kind+'_correct'] += correct
                    batch_events[kwargs.get('row', 0)] = dict(seconds=elapsed,
                        references_open=len(open_slots(meaning)), work=value.work.spent,
                        source_evidence=list(bindings(meaning).get('_source_evidence',(0.,0.))),
                        mode=meaning.mode, operations=operations,
                        query_results=[dict(frames=len(query.evidence.get('frames',())),
                            incomplete=query.evidence.get('incomplete',())) for query in checked_queries],
                        thought_steps=sum(record.kind == 'thought' for record in value.records),
                        trial_costs=list(model._last_thought_comparison['costs']),
                        trial_answer_costs=[item[1] for item in model._last_thought_comparison['components']],
                        trial_work_costs=list(model._last_thought_comparison.get('work_costs',(0.,0.))),
                        exhausted_closing=exhaustion)
                    if profile is not None:
                        (folder/'profile-context.json').write_text(json.dumps(dict(
                            seconds=elapsed, occupancy=len(model.symbolSpace.ltm_store),
                            capacity=model.symbolSpace.ltm_store.capacity,
                            **{key: value for key, value in batch_events[kwargs.get('row', 0)].items()
                               if key != 'seconds'}), indent=2)+'\n')
                    return value

                def batch(model, *args, **kwargs):
                    nonlocal batch_seconds
                    batch_events.clear()
                    timer = time.perf_counter()
                    try:
                        return ordinary_batch(model, *args, **kwargs)
                    finally:
                        batch_seconds = time.perf_counter()-timer

                def after(model, split, rows, output):
                    observer.after_batch(model, split, rows, output)
                    base = max(0., batch_seconds-sum(item['seconds'] for item in batch_events.values()))/len(rows)
                    sentences = []
                    for row, source in enumerate(rows):
                        address = data.source_addresses[split][source]
                        doc = data.math_chain_documents[split][address['document']]
                        text = doc.sentences[address['sentence']]
                        event = batch_events.get(row)
                        if event is not None and event['exhausted_closing'] is not None:
                            exhaustion = event['exhausted_closing']
                            examples = exhausted_examples[exhaustion['kind']]
                            if len(examples) < 10:
                                examples.append(dict(exhaustion, source=source, text=text))
                        is_question = address['sentence'] == doc.question
                        category = ('answer_line' if text.startswith('the answer is ') else
                            'question_with_episode' if is_question and event else
                            'question_without_episode' if is_question else 'plain')
                        seconds = base+(0. if event is None else event['seconds'])
                        counts[category] += 1
                        timings[category] += seconds
                        group = 'questions' if is_question else 'answer_lines' if category == 'answer_line' else 'declaratives'
                        counts[group+'_observed'] += 1
                        counts[group+'_open'] += event is not None and event['references_open'] > 0
                        counts[group+'_references_open'] += 0 if event is None else event['references_open']
                        counts[group+'_episodes'] += event is not None
                        if group == 'declaratives':
                            declaratives.append(event is not None)
                        if is_question and event is not None:
                            counts['question_cost_pairs'] += 1
                            counts['question_cost_nonties'] += event['trial_costs'][0] != event['trial_costs'][1]
                            counts['question_answer_nonties'] += event['trial_answer_costs'][0] != event['trial_answer_costs'][1]
                        sentences.append(dict(source=source, kind=category, text=text,
                                              seconds=seconds, episode=event))
                    counts['batches'] += 1
                    occupancy = len(model.symbolSpace.ltm_store)
                    batch_record = dict(batch=counts['batches'], seconds=batch_seconds,
                                        occupancy=occupancy, sentences=sentences)
                    batches.write(json.dumps(batch_record)+'\n')
                    batches.flush()
                    progress = dict(result, elapsed_seconds=time.perf_counter()-start,
                        counts=dict(counts), timing_seconds=dict(timings), occupancy=occupancy)
                    (folder/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(dict(batch=counts['batches'], total_sentences=sum(
                        counts[name] for name in ('plain', 'answer_line', 'question_with_episode', 'question_without_episode')),
                        seconds=batch_seconds, occupancy=occupancy)), flush=True)

                epoch_start = time.perf_counter()
                with patch.object(BasicModel, 'run_selected_thought', episode), patch.object(BasicModel, 'runBatch', batch):
                    report = present(model, split='train', optimizer=optimizer, batch_size=8, after_batch=after)
                result.update(status='completed', completed_epochs=1,
                    epoch_seconds=time.perf_counter()-epoch_start, report=report,
                    observer=observer.report(), occupancy=len(model.symbolSpace.ltm_store),
                    chooser_movement={name: float((value.detach()-initial[name]).norm())
                        for name, value in chooser.named_parameters() if name in initial},
                    first_epoch_questions=dict(observed=len(observer.questions),
                        references_left_open=counts['questions_open'], episodes_opened=counts['questions_episodes'],
                        bindings_correct=sum(row['bound_correct'] for row in observer.questions)))
                midpoint = len(declaratives)//2
                halves = (declaratives[:midpoint],declaratives[midpoint:])
                result['declarative_episode_halves'] = [dict(sentences=len(values), episodes=sum(values),
                    share=sum(values)/len(values)) for values in halves]
                result['development_certificate'] = dict(
                    question_answer_costs_nontied=counts['question_answer_nonties']>0,
                    chooser_moved=any(value>0 for value in result['chooser_movement'].values()),
                    declarative_episode_share_fell=result['declarative_episode_halves'][1]['share'] <
                        result['declarative_episode_halves'][0]['share'])
                result['development_certificate']['passed'] = all(result['development_certificate'].values())
                result['ordinary_exhaustion_certificate'] = dict(forced=False,
                    examples=exhausted_examples,
                    passed=all(counts['exhausted_'+kind+'_correct'] > 0 and
                               counts['exhausted_'+kind+'_observed'] == counts['exhausted_'+kind+'_correct']
                               for kind in ('declarative','question')))
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        # Development diagnosis only: preserve a failing native closing,
        # without changing the driver, observer, choices or failure outcome.
        try:
            tb = error.__traceback__
            while tb is not None:
                local = tb.tb_frame.f_locals
                if tb.tb_frame.f_code.co_name == '_commit_sentence' and 'view' in local:
                    row = local.get('b')
                    entries = local['view'].get('entries', ())
                    if type(row) is int and 0 <= row < len(entries):
                        torch.save(dict(program=entries[row], row=row,
                            semantic_extras=model.symbolSpace.ltm_store.semantic_extras()),
                            folder/'failed-closing.pt')
                tb = tb.tb_next
        except Exception as diagnostic_error:
            result['failure_capture_error'] = repr(diagnostic_error)
        raise
    finally:
        result.update(total_seconds=time.perf_counter()-start, counts=dict(counts),
            timing_seconds=dict(timings), seconds_per_sentence={key: value/counts[key] for key, value in timings.items()},
            source_unchanged=source == source_snapshot(ROOT))
        batches.close()
        if model is not None:
            try:
                torch.save(model.state_dict(),folder/'last-state.pt')
            except Exception as error:
                result['diagnostic_checkpoint_error'] = repr(error)
            model.End()
        (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], None if len(sys.argv)<4 else Path(sys.argv[3]))
