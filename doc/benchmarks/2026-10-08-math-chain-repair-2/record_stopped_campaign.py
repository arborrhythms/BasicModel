"""Read the stopped campaign's saved records; never resume or evaluate it."""
import hashlib
import json
from pathlib import Path

from finish_receipt import read, lines, summarize

HERE = Path(__file__).resolve().parent
OUT = HERE/'stopped-by-decision'


def main():
    decision = read(OUT/'decision.json')
    retained = read(OUT/'retained-files-sha256.json')
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
               for name,digest in retained.items())
    protocol = read(HERE/'protocol.json')
    runs, conditions, work, _ = summarize(HERE/'math-trainings',protocol)
    observed = []
    for row in runs:
        if row['status']=='not_started':
            continue
        row['status']='stopped_by_decision'
        row['held_out']=row['beyond_range']=None
        row['held_out_bar']=row['chain_bar']=None
        row['completed_for_evaluation']=False
        row['final_chooser_movement_available']=row['final_chooser_movement'] is not None
        row['movement_limitation']='No final parameter checkpoint was written before the decision; no movement is reconstructed from a rerun.'
        records=lines(HERE/'math-trainings'/row['name']/'questions.jsonl')
        epochs=[]
        for epoch in sorted({item['epoch'] for item in records}):
            items=[item for item in records if item['epoch']==epoch and item['phase']=='train']
            epochs.append(dict(epoch=epoch,epoch_completed=epoch<=row['completed_epochs'],
                questions=len(items),episodes=sum(item['episode'] for item in items),
                work=sum(item['work'] for item in items),
                correct_bindings=sum(item['bound_correct'] for item in items),
                correct_chains=sum(item['chain_correct'] for item in items),
                no_committed_binding=sum(item['committed_binding'] is None for item in items)))
        row['questions_by_epoch']=epochs
        row['observed_questions']=sum(item['questions'] for item in epochs)
        row['observed_correct_bindings']=sum(item['correct_bindings'] for item in epochs)
        observed.append(row)
    corrections=dict(status='deferred_not_implemented',authority='thinking spec sections 14.10–14.11; Alec, 2026-10-09',
        original_protocol='protocol.json',original_protocol_sha256=retained['protocol.json'],
        checkpoint='item 0 trained checkpoint, after runtime optimization under item 1',
        corrections=[
            dict(area='curriculum',required='One-step problems first: x is three. / what is x ?; then y is x; then plus zero, plus one, and longer chains.'),
            dict(area='answer_credit',required='Worked steps are intermediate answers. Each expected inference row is scored 0 when produced and 1 otherwise; A is the mean over those rows and the final binding. The sentence judgement remains R + A.'),
            dict(area='episode',required='After query finds candidates, the episode can bind a found candidate, or conclude with the best match bound.')],
        implementation_in_this_pass=False,new_measurement_declared=False,retries=0,replacements=0)
    (HERE/'protocol-corrections-deferred.json').write_text(json.dumps(corrections,indent=2)+'\n')
    summary=dict(**decision,unforced=True,completed_trainings=0,held_out_evaluation=None,
        beyond_range_evaluation=None,bar_claim=None,observations=observed,
        per_epoch_kind_report='episodes-by-start-epoch-kind.json',
        deferred_protocol='../protocol-corrections-deferred.json',
        retained_files_verified=len(retained),source_writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'episodes-by-start-epoch-kind.json').write_text(json.dumps(work,indent=2)+'\n')
    paragraphs=['# MM_math_chain stopped by decision — §14.10–§14.11','',
        'Stopped by Alec’s decision at **two of thirty trainings**: answer plus expectation completed **8 epochs**, expectation-only completed **9**. The partial next epochs are retained too. The remaining 28 never started. No attempt is retried or replaced. **No learning claim or held-out evaluation.**','',
        'The reason is the §14.10 reading: no correct binding in the first start; query/conclude drift as the store fills while the menu cannot bind a found candidate; the easiest corpus problem already needs coordinated steps; weak answer-cost separation for near-misses. The learning measurement is deferred to item 0 after runtime optimization. The three future corrections are recorded beside the untouched `protocol.json` in `protocol-corrections-deferred.json`; they are not implemented here.','',
        'All existing partial outcomes, logs and archives stay byte-for-byte as found. `retained-files-sha256.json` records the stop boundary. The campaign controller was already absent at that boundary; the two live workers were stopped explicitly. The old controller progress file is preserved, including its stale timestamp.','',
        '| Condition | Completed epochs | Observed questions, including partial next epoch | Correct bindings | First-epoch greedy what openings |',
        '|---|---:|---:|---:|---:|']
    for row in observed:
        greedy=row['greedy_what']
        paragraphs.append(f"| {row['condition']} | {row['completed_epochs']} | {row['observed_questions']} | {row['observed_correct_bindings']} | {greedy['first_epoch_what_opens']}/{greedy['first_epoch_questions_observed']} |")
    paragraphs+=['','These are **unforced observations**, not a bar. `summary.json` completes the per-start question/opening reports from saved logs; `episodes-by-start-epoch-kind.json` reports counts, share, work and mean work by epoch and sentence kind, marking partial epochs. Original files are not rewritten. Final chooser movement is unavailable because no final parameter checkpoint had been saved.','',
        '6.2 now closes on separately labelled **forced decomposition demonstrations** through the real driver, frozen observer and live explore suffix. The standing thirty remain as already measured. Stop for Claude’s review before any commit.']
    (OUT/'README.md').write_text('\n'.join(paragraphs)+'\n')
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
               for name,digest in retained.items())
    print(json.dumps(dict(stopped=True,epochs=decision['completed_epochs'],
        observed=[{key:row[key] for key in ('condition','observed_questions','observed_correct_bindings')} for row in observed],
        retained_files_verified=len(retained)),indent=2))


if __name__=='__main__':
    main()
