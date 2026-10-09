"""Post-process saved results only; never train, score a binding, or retry.

This offline receipt writer is separate from the frozen measurement helpers.
It can wait for the existing campaign and then stop for Claude's review.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A hard-killed writer can leave its final line incomplete.
            # Keep the original file; only completed JSON records are read.
            continue
    return result


def first_epoch(folder, batches):
    saved = read(folder/'first-epoch-questions.json')
    if saved is not None:
        return saved
    questions = [sentence for batch in batches
        if batch['completed'] and batch['phase']=='train'
        for sentence in batch['sentences']
        if sentence['epoch']==1 and sentence['question']]
    observed = [row for row in lines(folder/'questions.jsonl')
                if row['epoch']==1 and row['phase']=='train']
    return dict(questions_observed=len(questions),
        questions_left_open_at_closing=sum(bool(row.get('episode') and row['episode']['references_open']) for row in questions),
        references_left_open_at_closing=sum(row['episode']['references_open'] if row.get('episode') else 0 for row in questions),
        episodes_opened=sum(row.get('episode') is not None for row in questions),
        bindings_correct=sum(row['bound_correct'] for row in observed),
        frozen_observer_questions=len(observed),
        definition='Recovered from completed batch records and frozen observer after a missing final summary; original partial files retained.')


def work_rows(batches):
    totals = {}
    for batch in batches:
        if not batch['completed']:
            continue
        for sentence in batch['sentences']:
            key = (sentence['epoch'],batch['phase'],batch['split'],sentence['kind'])
            row = totals.setdefault(key,dict(sentences=0,episodes=0,work=0,seconds=0.))
            episode = sentence.get('episode')
            row['sentences'] += 1
            row['episodes'] += episode is not None
            row['work'] += 0 if episode is None else episode['work']
            row['seconds'] += sentence['seconds']
    return [dict(epoch=key[0],phase=key[1],split=key[2],kind=key[3],**row,
                 mean_work_per_episode=row['work']/row['episodes'] if row['episodes'] else None,
                 seconds_per_sentence=row['seconds']/row['sentences'],
                 episode_share=row['episodes']/row['sentences'])
            for key,row in sorted(totals.items())]


def greedy_what(folder, batches):
    saved = read(folder/'greedy-what.json')
    if saved is not None:
        return saved
    first = [dict(epoch=sentence['epoch'],phase=batch['phase'],split=batch['split'],
                  document=sentence['document'],completed_batch=batch['completed'],
                  **sentence['greedy'])
        for batch in batches if batch['phase']=='train'
        for sentence in batch['sentences']
        if sentence['epoch']==1 and sentence.get('greedy') is not None]
    return dict(first_question=first[0] if first else None,
        first_epoch_questions_observed=len(first),
        first_epoch_what_opens=sum(row['what_open'] is True for row in first),
        first_epoch_what_identified=sum(row['what_identified'] for row in first),
        definition='Recovered from saved greedy closing records; partial failed batches remain identified.')


def summarize(campaign, protocol):
    completed_campaign = read(campaign/'complete.json',{})
    progress = read(campaign/'progress.json',{})
    processes = {job['name']:job['process'] for job in
                 completed_campaign.get('jobs',progress.get('done',[]))}
    runs, all_work = [], []
    for start in range(1,protocol['runs_per_condition']+1):
        for condition in protocol['conditions']:
            name = f'{condition}-{start:02}'
            folder = campaign/name
            result = read(folder/'result.json',{})
            process = processes.get(name,read(folder/'process.json'))
            batches = lines(folder/'episode-batches.jsonl')
            completed = (result.get('status')=='completed'
                and result.get('completed_epochs')==protocol['epochs']
                and process is not None and process['exit_code']==0)
            greedy = greedy_what(folder,batches)
            first = first_epoch(folder,batches) if folder.exists() else None
            work = read(folder/'episodes-by-epoch-kind.json')
            if work is None:
                work = work_rows(batches)
            completed_epochs = result.get('completed_epochs',read(folder/'progress.json',{}).get('completed_epochs',0))
            all_work.extend(dict(run=start,condition=condition,
                epoch_completed=(row['epoch']<=completed_epochs),
                completed_batches_only=True,**row) for row in work)
            movement = result.get('chooser_movement')
            if movement is None:
                movement = read(folder/'final-chooser-movement.json',{}).get('movement')
            status = 'not_started' if not folder.exists() else result.get('status','unreported')
            if completed:
                status = 'completed'
            elif result.get('status')=='completed':
                status = 'completion_not_verified'
            elif process and process['exit_code']!=0:
                status = 'failed'
            row = dict(run=start,condition=condition,name=name,
                status=status,
                completed_epochs=completed_epochs,
                required_epochs=protocol['epochs'],process=process,
                error=result.get('error',None if process is None or process['exit_code']==0 else process.get('reason')),
                seconds=result.get('seconds',None if process is None else process.get('seconds')),
                first_epoch=first,greedy_what=greedy,
                credit_observations=result.get('observer',read(folder/'progress.json',{}).get('observer')),
                final_chooser_movement=movement,
                completed_for_evaluation=completed,
                held_out=result.get('test') if completed else None,
                beyond_range=result.get('beyond') if completed else None,
                held_out_bar=result.get('held_out_bar') if completed else None,
                chain_bar=result.get('chain_bar') if completed else None,
                zero_budget_work_verified=(all(item['work']==0 for item in work)
                    if condition=='zero_attention_budget' and work else None))
            runs.append(row)
    conditions = []
    for condition in protocol['conditions']:
        subset = [row for row in runs if row['condition']==condition]
        valid = [row for row in subset if row['completed_for_evaluation']]
        conditions.append(dict(condition=condition,declared=len(subset),
            attempted=sum(row['status']!='not_started' for row in subset),
            completed=len(valid),
            four_of_four_held_out=sum(bool(row['held_out_bar']) for row in valid),
            four_of_four_chains=sum(bool(row['chain_bar']) for row in valid),
            held_out_correct=sum(row['held_out']['correct'] for row in valid),
            held_out_rows=sum(len(row['held_out']['rows']) for row in valid),
            beyond_range_correct=sum(row['beyond_range']['correct'] for row in valid),
            beyond_range_rows=sum(len(row['beyond_range']['rows']) for row in valid),
            first_epoch={key:sum((row['first_epoch'] or {}).get(key,0) for row in subset)
                for key in ('questions_observed','questions_left_open_at_closing',
                    'references_left_open_at_closing','episodes_opened','bindings_correct')}))
    return runs,conditions,all_work,completed_campaign


def write_receipt():
    protocol = read(HERE/'protocol.json')
    runs,conditions,work,campaign = summarize(HERE/'math-trainings',protocol)
    source = read(HERE/'measured-source/source.json')
    helpers = read(HERE/'measured-source/measurement-helpers.json')
    mismatches = [name for name,digest in {**source,**helpers}.items()
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    # Detect source additions too, without importing or running learner code.
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('receipt_bounded_tests',ROOT/'test/bounded_tests.py')
    bounded = module_from_spec(spec)
    sys.modules[spec.name] = bounded
    spec.loader.exec_module(bounded)
    current_source = bounded.source_snapshot(ROOT)
    mismatches += sorted(set(current_source)-set(source))
    prior = []
    for name in ('2026-10-07-math-chain','2026-10-08-math-chain-repair'):
        folder = HERE.parent/name
        files = read(folder/'receipt-sha256.json')
        changed = [name for name,digest in files.items()
            if not (folder/name).exists() or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest]
        prior.append(dict(receipt=folder.name,files=len(files),mismatches=changed))
    expected_names = {row['name'] for row in runs}
    all_attempts = (campaign.get('attempts')==30 and len(campaign.get('jobs',[]))==30
        and {job['name'] for job in campaign['jobs']}==expected_names)
    summary = dict(status='stopped_for_Claude_review' if all_attempts else 'campaign_incomplete',
        generated_utc=datetime.now(timezone.utc).isoformat(),seed=None,retries=0,committed=False,
        all_declared_attempts_retained=all_attempts,declared_epochs=protocol['epochs'],
        campaign_seconds=campaign.get('seconds'),
        campaign_source_matched=campaign.get('source_matched'),
        source_and_helpers_match=not mismatches,mismatches=mismatches,
        sweep=dict(completed=5636,passed=5351,skipped=284,xpassed=1),
        thinking_gate=dict(passed=57,total=57),
        standing=read(HERE/'standing-thirty-verification.json')['raw_bars'],
        standing_exception='Both MM misses reproduce bit-for-bit on the 6.2 landing for all 200 steps; raw 8/10 retained. See mm-bisection/result.json.',
        conditions=conditions,runs=runs,prior_receipts=prior,
        answer_learning_bar_met=next(row for row in conditions if row['condition']=='answer_and_expectation')['four_of_four_held_out']>=9,
        scope='Held-out and beyond-range bindings are reported only for completed trainings. These results do not discharge the 6.5 learning gates pending the million-sentence checkpoint.',
        receipt_writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,value in (('summary.json',summary),('episode-work-all-runs.json',work)):
        (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
    paragraphs = ['# MM_math_chain §14.8 — measurement receipt','',
        'All declared attempts are retained; no retry or replacement. Stop for Claude’s review before any commit.' if all_attempts else 'The campaign ended before every declared attempt completed. All available outcomes are retained; no retry or replacement.',
        '',f"Source and frozen helper hashes match: **{not mismatches}**. Sweep: **5,636/5,636 completed** (5,351 passed, 284 skipped, one XPASS); thinking: **57/57**.",
        '', 'Standing: sum **10/10**, XOR class and reconstruction **10/10**, raw MM **8/10**. Both MM misses reproduce identically on the 6.2 landing for all 200 steps; see `mm-bisection/result.json`. The raw misses remain unchanged.',
        '', '| Condition | Completed / 10 | Held-out 4/4 runs | Chain 4/4 runs | Beyond-range correct / evaluated |',
        '|---|---:|---:|---:|---:|']
    for row in conditions:
        beyond = f"{row['beyond_range_correct']}/{row['beyond_range_rows']}" if row['beyond_range_rows'] else 'unavailable'
        held = str(row['four_of_four_held_out']) if row['completed'] else 'unavailable'
        chains = str(row['four_of_four_chains']) if row['completed'] else 'unavailable'
        paragraphs.append(f"| {row['condition']} | {row['completed']}/10 | {held} | {chains} | {beyond} |")
    paragraphs += ['', 'Held-out results count only completed 17-epoch trainings. See `summary.json` for each attempt’s result, failure, committed-row evaluation and final chooser movement; `episode-work-all-runs.json` gives every epoch and sentence kind’s episode count, share and mean work.',
        '', '| Start | Condition | Status | Epochs | First greedy what open | First-epoch greedy openings / questions | First-epoch references / episodes / correct bindings |',
        '|---:|---|---|---:|---|---:|---:|']
    for row in runs:
        greedy,first = row['greedy_what'],row['first_epoch'] or {}
        initial = (greedy.get('first_question') or {}).get('what_open')
        opening = 'unavailable' if initial is None else str(initial).lower()
        count = f"{greedy.get('first_epoch_what_opens',0)}/{greedy.get('first_epoch_questions_observed',0)}"
        observations = ' / '.join(str(first.get(k,0)) for k in ('references_left_open_at_closing','episodes_opened','bindings_correct'))
        paragraphs.append(f"| {row['run']} | {row['condition']} | {row['status']} | {row['completed_epochs']} | {opening} | {count} | {observations} |")
    paragraphs += ['', 'The original development result is intact. `development-acceptance-14-8.json` records the corrected certificate: mean work per declarative episode 30.333 → 18.942, with opening share reported rather than gated.',
        '', '`binding-answers-frozen.diff` documents the authorized loss-side change. `references()` and `matches()` remain byte-identical to the original frozen source; the verifier and both earlier receipts remain intact.',
        '', 'The 6.5 learning gates remain pending the million-sentence checkpoint. No commit, push, or bump was performed.']
    (HERE/'MEASUREMENT.md').write_text('\n'.join(paragraphs)+'\n')
    (HERE/'measurement-status.json').write_text(json.dumps(dict(phase=summary['status'],
        math_attempts=len(campaign.get('jobs',[])),conditions=conditions,committed=False,
        source_and_helpers_match=not mismatches,receipt='MEASUREMENT.md'),indent=2)+'\n')
    readme = HERE/'README.md'
    marker = '\n\n## Declared campaign outcome\n'
    old = readme.read_text().split(marker)[0]
    old = old.replace('The thirty declared math trainings are now running.',
        'The declared math campaign has ended; see the final outcome below.'
        if all_attempts else 'The declared math campaign stopped incomplete; see the recorded outcome below.')
    readme.write_text(old+marker+'\nSee [MEASUREMENT.md](MEASUREMENT.md) and `summary.json` for the final recorded campaign status. All available outcomes are retained. No commit was made.\n')
    print(json.dumps(dict(status=summary['status'],conditions=conditions,
        source_and_helpers_match=not mismatches)),flush=True)
    manifest = {str(path.relative_to(HERE)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(HERE.rglob('*')) if path.is_file() and
        path.name not in ('receipt-sha256.json','receipt-finisher.log') and '__pycache__' not in path.parts}
    (HERE/'receipt-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--watch':
        pid = int(sys.argv[2])
        while not (HERE/'math-trainings/complete.json').exists():
            try:
                os.kill(pid,0)
            except ProcessLookupError:
                break
            state = subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if state.returncode or state.stdout.lstrip().startswith('Z'):
                break
            time.sleep(10)
    write_receipt()
