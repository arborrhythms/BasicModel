"""Measure native admission through whole training documents at a fitted size.

An exhausted admission is a failed measurement with the last complete document,
not a resized or partially trained model. A fresh selected run uses that prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bin'))
from util import init_config,TheXMLConfig,init_compile_backend
from data import TheData
from Models import BaseModel, BasicModel
from Corpus import admit_training_forms
from Spaces import _concept_alloc_of
from corpus_audit import peak_bytes


def admit_prefix(config, output):
    from types import SimpleNamespace
    args=SimpleNamespace(config=Path(config),output=Path(output))
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    init_compile_backend('none')
    init_config(path=str(args.config))
    cfg=TheXMLConfig.data['architecture']['data']
    began=time.monotonic()
    TheData.load(cfg['dataset'],dat=cfg,max_docs=cfg['maxDocs'],max_sentence_words=cfg['maxSentenceWords'])
    model=BaseModel.from_config(str(args.config),data=TheData)[0]
    completed=-1;sentences=0;attempted=None
    def occupancy():
        owner=model._concept_owner();alloc=_concept_alloc_of(owner)
        identity=model.perceptualSpace.percept_store.identity
        return dict(percept=len(model.perceptualSpace.percept_store),forms=len(identity.word_rows),
                    conceptual_next=alloc.next_id,relation_highest=int(alloc.parts.count)-1)
    try:
        rows=TheData.source_addresses['train'];lo=0
        while lo<len(rows):
            document=rows[lo]['document'];hi=lo+1
            while hi<len(rows) and rows[hi]['document']==document:hi+=1
            attempted=dict(document=document,start=lo,end=hi,address=rows[lo])
            save('attempted.json',dict(**attempted,occupied=occupancy()))
            answers=(TheData.text_answers or {}).get('train',())
            admit_training_forms(model,(TheData.train_input[i] for i in range(lo,hi)),
                (answers[i] for i in range(lo,hi) if i<len(answers)))
            completed=document;sentences=hi
            row=dict(document=document,training_sentences=sentences,occupied=occupancy(),
                     elapsed_seconds=time.monotonic()-began,peak_bytes=peak_bytes())
            with (args.output/'documents.jsonl').open('a') as target:target.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            lo=hi
        report=dict(status='complete',max_docs=cfg['maxDocs'])
    except Exception as error:
        # The caller must inspect the concrete failure. Only a proved native
        # capacity exhaustion justifies selecting the preceding whole prefix.
        report=dict(status='failed',error=repr(error),error_type=type(error).__name__,
                    error_message=str(error),attempted=attempted,
                    last_completed_document=completed,last_completed_training_sentences=sentences,
                    largest_completed_max_docs=completed+1)
        save('result.json',dict(**report,occupied=occupancy(),
            elapsed_seconds=time.monotonic()-began,peak_bytes=peak_bytes()))
        raise
    else:
        audit=model.perceptualSpace.percept_store.identity.audit()
        if any(audit[key] for key in ('collisions','reconstruction_errors','containment_violations')):
            raise RuntimeError('admitted native prefix failed its identity audit')
        save('identity-audit.json',audit)
        save('result.json',dict(**report,training_sentences=sentences,occupied=occupancy(),
            corpus=TheData.source_manifest,elapsed_seconds=time.monotonic()-began,peak_bytes=peak_bytes()))
    finally:
        model.End()


def selected_document_count(report, guard, requested):
    """Only known admission exhaustion selects the previous whole document."""
    if guard['status'] == 'complete' and report['status'] == 'complete':
        return requested
    message = report.get('error_message', '')
    expected = (
        report.get('error_type') == 'RuntimeError' and (
            message.startswith('RadixLayer percept codebook capacity exhausted: ')
            or message in (
                'definition store capacity exhausted; no word was admitted',
                'concept row capacity exhausted before word admission; no identity was minted'))
        or report.get('error_type') == 'OverflowError'
        and message == 'LTM2 symbol capacity exhausted before allocation')
    if guard['status'] != 'failed' or not expected:
        raise RuntimeError(f'admission failed outside a native inventory limit: {guard}, {report}')
    count = report['largest_completed_max_docs']
    if not 0 < count < requested:
        raise RuntimeError(f'no nonempty whole-document prefix: {report}')
    return count


def train_prefix(config, output):
    """Native initial, periodic and final checkpoints; one complete epoch."""
    import os
    from Corpus import file_digest
    from corpus_capacity import train_brick
    output = Path(output)
    init_compile_backend('none')
    os.environ['BASIC_WEIGHTS_PATH'] = str((output / 'periodic.ckpt').resolve())
    native = BasicModel.runEpoch
    begun = False

    def with_initial_checkpoint(model, *args, **kwargs):
        nonlocal begun
        if begun:
            raise RuntimeError('the declared prefix run contains one epoch')
        begun = True
        target = output / 'initial.ckpt'
        model.save_weights(str(target))
        (output / 'initial-checkpoint.json').write_text(json.dumps(dict(
            path=str(target.resolve()), sha256=file_digest(target),
            cursor=getattr(model, '_checkpoint_corpus_cursor', None)), indent=2) + '\n')
        return native(model, *args, **kwargs)

    BasicModel.runEpoch = with_initial_checkpoint
    try:
        return train_brick(config, output, complete_prefix=True)
    finally:
        BasicModel.runEpoch = native


def run_pipeline(root, output):
    """Fresh fixed-grid search, admission, revalidation, then the large prefix."""
    import xml.etree.ElementTree as ET
    from bounded_tests import source_snapshot
    from corpus_capacity import guarded_trial, search
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = source_snapshot(root)
    (output / 'source.json').write_text(json.dumps(frozen, indent=2) + '\n')
    grid = root / 'data/corpus-capacity-grid.json'
    declaration = json.loads(grid.read_text())
    config = root / 'data/BasicModel_corpus.xml'
    result = search(config, grid, output / 'capacity')
    if result['status'] != 'complete' or not result['winner']:
        raise RuntimeError(f'no fully revalidated capacity fit: {result["status"]}')
    selected = output / 'selected.xml'
    tree = ET.parse(result['winner_config'])
    data = tree.getroot().find('architecture/data')
    requested = int(data.find('maxDocs').text)
    tree.write(selected, encoding='unicode')

    def worker(mode, name, timeout):
        command = [sys.executable, str(Path(__file__).resolve()), mode,
                   '--config', str(selected.resolve()),
                   '--output', str((output / name).resolve())]
        return guarded_trial(command, output / (name + '-guard'),
            memory_bytes=declaration['memory_bytes'], timeout=timeout)

    admission = worker('--admit-prefix', 'admission', declaration['timeout'])
    report_path = output / 'admission/result.json'
    if not report_path.exists():
        raise RuntimeError(f'admission produced no result: {admission}')
    documents = selected_document_count(json.loads(report_path.read_text()), admission, requested)
    data.find('maxDocs').text = str(documents)
    tree.write(selected, encoding='unicode')
    # Prefix selection changes the loaded data; validate this exact config afresh.
    validation = worker('--train-brick', 'selected-brick', declaration['timeout'])
    if validation['status'] != 'complete':
        raise RuntimeError(f'selected-prefix B28/W256 validation failed: {validation}')
    training = worker('--train-prefix', 'training', 86400)
    if training['status'] != 'complete':
        raise RuntimeError(f'selected-prefix training failed: {training}')
    if source_snapshot(root) != frozen:
        raise RuntimeError('large-corpus source changed during measurement')
    report = dict(status='complete', documents=documents, winner=result['winner'],
        config=str(selected), training=json.loads((output / 'training/result.json').read_text()))
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--admit-prefix', action='store_true')
    mode.add_argument('--train-prefix', action='store_true')
    mode.add_argument('--train-brick', action='store_true')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.admit_prefix:
        admit_prefix(args.config, args.output)
    elif args.train_prefix:
        train_prefix(args.config, args.output)
    else:
        from corpus_capacity import train_brick
        init_compile_backend('none')
        train_brick(args.config, args.output)

