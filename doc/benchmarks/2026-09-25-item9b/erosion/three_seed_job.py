"""Item 9b's descriptive, three-seed native-membership erosion measurement.

Seeds make measurements reproducible; no ordering is asserted or tuned.
The probe labels are XOR outcomes and native orthographic property classes,
not hand-assigned semantic categories. See the emitted protocol and receipt.
"""
import argparse
import copy
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch


@contextmanager
def eager_native_loops():
    """Execute the native recurrence bodies without per-vocabulary compilation.

    This CPU measurement compares memberships and learning, not throughput.
    Carry tensors and operations are identical; only the loop dispatcher is
    eager so changing sentence banks do not compile hundreds of graphs.
    """
    original = torch.while_loop
    def run_loop(cond, body, carries):
        while bool(cond(*carries)):
            carries = body(*carries)
        return carries
    torch.while_loop = run_loop
    try:
        yield
    finally:
        torch.while_loop = original


def discrimination(readings, labels):
    """Mean pairwise L2 distances, with every unordered pair counted once."""
    within, between = [], []
    for i, left in enumerate(readings):
        for j in range(i + 1, len(readings)):
            distance = float(torch.linalg.vector_norm(left - readings[j]))
            (within if labels[i] == labels[j] else between).append(distance)
    if not within or not between:
        raise ValueError('erosion probes need both within- and between-category pairs')
    dw, db = sum(within) / len(within), sum(between) / len(between)
    return dict(cp=db-dw, within=dw, between=db,
                within_pairs=len(within), between_pairs=len(between))


def launch_probes(root):
    from data import iter_documents
    from util import parse
    shard = root / 'data/fineweb/shard_00000.parquet'
    documents = list(iter_documents([str(shard)], max_docs=20))
    if len(documents) != 20:
        raise RuntimeError('the erosion gate requires the 20-document launch corpus')
    sentences, words = [], []
    for document in documents:
        candidates = [text for text, _ in parse(document, lex='sentences')
                      if 1 <= len(text.split()) <= 12 and len(text.encode()) <= 256]
        if not candidates:
            raise RuntimeError('launch document has no complete sentence within the declared probe budget')
        sentences.append(candidates[0])
        # Fixed discovery order, up to four distinct alphanumeric word forms
        # per document, retaining the native letter/digit/case boundaries.
        tokens = [word for word, _ in parse(document, lex='words') if word.isalnum()]
        words.extend(list(dict.fromkeys(tokens))[:4])
    return sentences, list(dict.fromkeys(words)), dict(
        shard=str(shard.relative_to(root)),
        document_sha256=[hashlib.sha256(d.encode()).hexdigest() for d in documents],
        sentences=sentences)


def make_config(root, destination):
    tree = ET.parse(root / 'data/BasicModel.xml')
    r = tree.getroot()
    def put(path, value):
        parent, tag = path.rsplit('/', 1)
        node = r.find(path)
        if node is None: node = ET.SubElement(r.find(parent), tag)
        node.text = str(value).lower() if isinstance(value, bool) else str(value)
    for section in ('InputSpace', 'PartSpace', 'ConceptualSpace', 'WholeSpace', 'OutputSpace'):
        for tag in ('nDim', 'nInputDim', 'nOutputDim'):
            if r.find(f'{section}/{tag}') is not None: put(f'{section}/{tag}', 16)
    put('InputSpace/nOutput', 512)
    put('PartSpace/nInput', 512)
    put('PartSpace/nVectors', 4096)
    put('ConceptualSpace/nVectors', 2048)
    put('ConceptualSpace/activeVectors', 2048)
    put('OutputSpace/nInputDim', 16)
    put('OutputSpace/nDim', 16)
    for tag, value in dict(serial=True, modeSchedule='serial', serialWordCapacity=32,
            serialWordBuckets='32', subsymbolicOrder=1, symbolicOrder=3,
            transformChooserDepth=1, answerSynthesis=False, attentionPromotion=True,
            conceptualPi=True).items(): put('architecture/'+tag, value)
    for tag,value in dict(autoload=False, autosave=False, checkpointEveryBatches=0,
            numWorkers=0, batchSize=1, reconstructionPlacement='eager',
            intraLossWeight=0., forwardGrammarWeight=0., branchDiagnosticsEvery=0).items():
        put('architecture/training/'+tag,value)
    put('architecture/data/dataset','text')
    put('architecture/weightsPath', destination.parent/'unused.ckpt')
    tree.write(destination, encoding='unicode')
    return destination


def read_probes(model, probes):
    from ModeSchedule import ModeSchedule
    from Spaces import _concept_alloc_of
    cs = model._concept_owner()
    readings = {False: [], True: []}
    active, learned = [], []
    previous = [(sp, getattr(sp, '_online_learning_frozen', False)) for sp in model.spaces]
    for sp,_ in previous: sp._online_learning_frozen = True
    try:
        with ModeSchedule('parallel').parallel_pass(model), torch.no_grad():
            for text in probes:
                raw = model.inputSpace.prepInput([text])
                state = model.understand(raw)
                field = state.reconstruction_carriers['field']
                evidence = field.evidence
                rows = model._combine_last_cs_sub._concept_inventory_rows
                if rows.ndim == 2: rows=rows[:,0]
                # The unmasked read is the actual reverse-pi symbol query.
                # It cannot create an interpretation or assert unseen values.
                words = [cid for cid in cs.word_concepts(text)
                         if cs._concept_source_order(cid) == 0]
                chosen = torch.zeros_like(evidence)
                ids = field.concept_ids[:,0] if field.concept_ids.ndim == 2 else field.concept_ids
                for word in words:
                    chosen[ids == word] = evidence[ids == word]
                feedback = cs.cs_reverse_presence(chosen, observed=evidence, inventory_rows=rows)
                active.append(int(evidence.count_nonzero()))
                learned.append(bool(words))
                for masked, values in ((True,evidence),(False,feedback)):
                    vector = torch.zeros(int(cs.nVectors), 2)
                    valid = torch.tensor([row >= 0 and cs._order0_inventory_row(int(row))
                                          for row in rows.tolist()])
                    vector[rows[valid].cpu()] = values[valid,0].amax(1).cpu()
                    readings[masked].append(vector.flatten())
                model.End()
    finally:
        for sp,value in previous: sp._online_learning_frozen=value
    return readings, dict(nonzero_readings=active, lexical_coverage=sum(learned), probes=len(probes))


@eager_native_loops()
def run(destination, seeds=(0,1,2)):
    import Language
    from Models import BaseModel
    from ModeSchedule import ModeSchedule
    from data import TheData
    from util import init_config, init_device, init_compile_backend
    from What import What
    root=Path(__file__).resolve().parents[1]
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    init_device('cpu'); init_compile_backend('none'); torch.set_num_threads(1)
    sentences, words, provenance=launch_probes(root)
    config=make_config(root,destination/'model.xml')
    init_config(str(config),defaults_path=str(root/'data/model.xml'))
    TheData.load('text',num_shards=1,max_docs=20,
                 shard_dir=str(root/'data/fineweb'),max_sentence_words=32)
    smoke=['00','01','10','11']; training=smoke+sentences
    report=dict(protocol=dict(seeds=list(seeds),schedules=['serial','parallel','interleave:2'],
        training=training, epochs=1, probe_scope='XOR plus native orthographic property classes; no semantic-category claim',
        initialization='one model per seed, deep-copied before each condition',
        execution='CPU; native tensor recurrence bodies with an eager loop dispatcher; no throughput claim',
        symbol_mask='native field with symbols offline versus the owned word-symbol reverse-pi query',
        gate='all-three-seed ordering only; every violation is retained'), corpus=provenance, conditions=[])
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        Language.TheGrammar._configured=False
        base,_=BaseModel.from_config(str(config),data=TheData)
        base.set_sigma(0); base.checkpoint_every_batches=0
        base._tensor_peer_while_eager=True
        base._chart_compose_per_word=lambda:None
        base.reconstruction_placement='eager'
        # FineWeb classes are the native property identity tuples.
        labels=[tuple(base.wholeSpaces[0].property_rows_for_bytes(word)) for word in words]
        for schedule in report['protocol']['schedules']:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model=copy.deepcopy(base); model.mode_schedule=ModeSchedule(schedule)
            optimizer=model.getOptimizer(lr=.001)
            phases=[]; start=time.perf_counter()
            context=(model.mode_schedule.parallel_pass(model) if schedule=='parallel' else nullcontext())
            with context:
                for step,text in enumerate(training):
                    raw=model.inputSpace.prepInput([text])
                    result,_=model.runBatch(train=True,batchNum=step,batchSize=1,optimizer=optimizer,
                        split='train', batch_override=(raw,torch.empty(1,0)),
                        questions=(What.present(0,split='train'),))
                    phases.append({name: float(value) for name,value in model.primary_costs().items()
                                   if torch.is_tensor(value)})
                    # Each selected sentence is one completed document probe,
                    # using the ordinary runEpoch boundary and learning hooks.
                    model.dispatch_per_row_reset([True])
                    model.dispatch_soft_reset()
                    model.post_tick_compact()
                    (destination/'progress.json').write_text(json.dumps(dict(
                        seed=seed, schedule=schedule, completed_sentences=step+1)))
            metrics={}
            for name, probes, classes in (('smoke',smoke,[0,1,1,0]),('fineweb',words,labels)):
                reads,coverage=read_probes(model,probes)
                metrics[name]={'coverage':coverage,'probes':probes,'classes':classes}
                for masked, vectors in reads.items():
                    m=discrimination(vectors,classes)
                    # Alternatives are structural, not a score-threshold count.
                    from Spaces import _concept_alloc_of
                    store=_concept_alloc_of(model._concept_owner()).layer()
                    admitted = {row for row in range(store.nOutput)
                        if model._concept_owner()._order0_inventory_row(row)
                        and model._concept_owner().concept_id_at_row(row) is not None
                        and not bool(store.provisional[row])}
                    by_target={}
                    for (target,col),pos in store._index.items():
                        if target in admitted and float(store.values[pos])>0:
                            by_target.setdefault(target,set()).add(col)
                    m['admitted_order0_concepts'] = len(admitted)
                    m['alternatives_per_concept']=sum(map(len,by_target.values()))/max(1,len(admitted))
                    m['concepts_with_alternatives']=len(by_target)
                    metrics[name]['masked' if masked else 'unmasked']=m
            row=dict(seed=seed,schedule=schedule,seconds=time.perf_counter()-start,
                serial_training_steps=int(model._training_step_count),
                parallel_passes=model.mode_schedule.completed_parallel,losses=phases,metrics=metrics)
            report['conditions'].append(row)
            (destination/'results.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(seed=seed,schedule=schedule,seconds=row['seconds'],metrics=metrics)),flush=True)
            del model,optimizer;gc.collect()
        del base;gc.collect()
    findings=[]
    for workload in ('smoke','fineweb'):
        for seed in seeds:
            c={r['schedule']:r['metrics'][workload] for r in report['conditions'] if r['seed']==seed}
            tests={'parallel_cp_below_serial':c['parallel']['masked']['cp']<c['serial']['masked']['cp'],
                   'parallel_within_above_serial':c['parallel']['masked']['within']>c['serial']['masked']['within'],
                   'parallel_alternatives_above_serial':c['parallel']['masked']['alternatives_per_concept']>c['serial']['masked']['alternatives_per_concept'],
                   'interleave_cp_at_least_parallel':c['interleave:2']['unmasked']['cp']>c['parallel']['unmasked']['cp'],
                   'interleave_within_above_serial':c['interleave:2']['unmasked']['within']>c['serial']['unmasked']['within']}
            tests['interleave_cp_nearer_serial_than_parallel'] = (
                abs(c['interleave:2']['unmasked']['cp'] - c['serial']['unmasked']['cp'])
                <= abs(c['parallel']['unmasked']['cp'] - c['serial']['unmasked']['cp']))
            effects = {mode: v['unmasked']['cp'] - v['masked']['cp'] for mode,v in c.items()}
            tests['mask_effect_largest_in_serial'] = all(
                effects['serial'] > effect for mode,effect in effects.items() if mode != 'serial')
            for mode,v in c.items(): tests['mask_lowers_cp_'+mode]=v['masked']['cp']<v['unmasked']['cp']
            findings.append(dict(workload=workload,seed=seed,orderings=tests))
    report['findings']=findings
    (destination/'results.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.out)
