"""One NP lesson from its predecessor artifact; stop at a failed gate.

The fixed factorial is partitioned into a prerequisite sequence. Each case
is trained twice, or held out once, in exactly one lesson. Passing the prior
lesson is required before the next introduces an unknown construction.
"""
import argparse
from contextlib import nullcontext
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

from np_corpus import DEST, load
from constituent_budget import paired_updates_only
from constituent_measurement import observe, grade, summarize
from np_trajectory import NPTrajectory
from identity_experiment import make_model, restore_warmup, reset, save, digest

SEED, RATE, EPOCHS = 20261010, .003, 2
MENU = ('STOP', 'lower', 'definite', 'intersection', 'product')
# Learn modification on two familiar words before asking the reader to
# schedule it inside a larger NP. Nesting is then the next unknown.
STAGES = ('lower', 'definite', 'description', 'adjective', 'definite_adjective')


def first_lesson(row):
    return lesson_stage(row) == 'lower'


def lesson_stage(row):
    if row['id'].split(':')[1] == 'description':
        return 'description'
    words=row['text'].split()
    if len(words)==2:
        return 'lower' if words[0]=='a' else 'definite'
    return 'adjective' if words[0]=='a' else 'definite_adjective'


def batches(rows, rng=None):
    """Bare NPs, adjective NPs, descriptions; counterbalanced within family."""
    for family in ('bare', 'adjective', 'description'):
        ids = [i for i,row in enumerate(rows) if
               ('description' if row['id'].split(':')[1]=='description' else
                'bare' if len(row['text'].split())==2 else 'adjective') == family]
        if rng is not None:
            ids = list(map(int,rng.permutation(ids)))
        for offset in range(0,len(ids),4):
            yield ids[offset:offset+4]


def tables(model, forms):
    from Language import TheGrammar
    from PartsOfSpeech import operation_name, flow_mask
    owner, parts = model._concept_owner(), model.languageSpace.parts_of_speech
    word_rows, noun_rows = [], []
    for form in sorted(forms):
        word = owner.definitions.word(form=form)
        word = -1 if word is None else int(word)
        concepts = [] if word < 0 else list(map(int,owner.definitions.objects(word)))
        individuals = [cid for cid in concepts if cid in owner.components.nouns.ids]
        word_rows.append(dict(word=form, identity=word, part_of_speech=parts.part_of_speech(word),
                              counts=dict(parts.counts.get(word,{})),observed=word in parts.counts))
        noun_rows.append(dict(word=form, identity=word, concepts=concepts,
                              individual_columns=individuals, noun_prior=bool(individuals)))
    head_pairs,flow_pairs=set(),set()
    for rule in TheGrammar.rules_upward:
        name=operation_name(rule)
        if not name:
            continue
        mask=flow_mask(rule)
        roles=(2,1) if len(rule.lhs.split(','))==2 else ((1 if mask==1 else 2 if mask==2 else 0),)
        if len(roles)==1:
            flow_pairs.update(frozenset((f'{name}_argin{i+1}',f'{name}_argout1'))
                              for i in range(len(rule.rhs_symbols)) if mask & (1<<i))
        for output, role in enumerate(roles,1):
            if role:
                head_pairs.add(frozenset((f'{name}_argin{role}',f'{name}_argout{output}')))
    collapse=parts.collapse()
    for row in collapse:
        row['same_head_by_construction']=frozenset((row['left'],row['right'])) in head_pairs
        row['flow_related_by_construction']=frozenset((row['left'],row['right'])) in flow_pairs|head_pairs
    modification_flow={}
    for name in ('intersection','product'):
        slots=tuple(f'{name}_{slot}' for slot in ('argin1','argin2','argout1'))
        occupants={slot:{str(word):counts[slot] for word,counts in parts.counts.items() if counts.get(slot)} for slot in slots}
        combined={word:sum(occupants[slot].get(word,0) for slot in slots[:2])
                  for word in set(occupants[slots[0]])|set(occupants[slots[1]])}
        modification_flow[name]=dict(occupants=occupants,combined_inputs=combined,
            output_equals_combined_inputs=occupants[slots[2]]==combined,
            observed=bool(combined),by_construction=name=='intersection',
            declared_flow='all operands' if name=='intersection' else 'empty')
    return dict(words=word_rows, collapse=collapse, noun_prior=noun_rows,
        intersection_flow=modification_flow['intersection'],product_flow=modification_flow['product'])



def summarize_stage(results,stage):
    """Gate behaviors (§5a), and report the operators and description stamps."""
    summary=summarize(results)
    gates=summary['gates']
    reports={name:gates.pop(name) for name in ('reductions','unlowered')}
    # These are observations, not requirements for a particular reading.
    for report in reports.values():
        for field in ('passed','threshold','requirement'):
            report.pop(field,None)
    correct=sum(row['brackets'] for row in results)
    gates['brackets'].update(correct=correct,total=len(results),
        passed=bool(results) and correct/len(results)>=.9)
    determiner=[row for row in results if row['nps']]
    if determiner:
        gates['occurrences']=summarize(determiner)['gates']['occurrences']
    else:
        gates.pop('occurrences')
    questions=[row for row in results if row.get('question_expected', 'answer_exact' in row)]
    if questions:
        correct=sum(row.get('answer_exact',False) for row in questions)
        gates['answers']=dict(correct=correct,total=len(questions),threshold=1.,passed=correct==len(questions))
    from collections import Counter
    reports['kept_programs']=dict(Counter(' → '.join(row.get('observed',{}).get('program',[])) for row in results))
    summary.update(reports=reports,passed=all(gate['passed'] for gate in gates.values()))
    return summary


def run(args):
    from Models import BaseModel
    from What import What
    from AttentionLesson import prompt_need
    from bounded_tests import source_snapshot
    restore = BaseModel.load_weights
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train,_ = load('train')
    all_training=train
    evaluation,labels = load('eval')
    lessons = json.loads((DEST/'train.lessons.json').read_text())
    questions = {split:json.loads((DEST/f'{split}.questions.json').read_text()) for split in ('train','eval')}
    assert len(train)==len(lessons)==42 and len(evaluation)==14
    final=args.stage=='final'
    selected=[i for i,row in enumerate(train) if lesson_stage(row)==args.stage]
    train,lessons=[train[i] for i in selected],[lessons[i] for i in selected]
    evaluation=[row for row in evaluation if final or lesson_stage(row)==args.stage]
    training_presentations=EPOCHS*len(train)
    paired_updates=2*EPOCHS*len(list(batches(train)))
    previous=None
    if args.stage!='lower':
        previous=json.loads((args.warmup.parent/'results.json').read_text())
        predecessor=STAGES[-1] if final else STAGES[STAGES.index(args.stage)-1]
        expected_ids=[row['id'] for row in all_training if lesson_stage(row)==predecessor]
        if (not previous['passed'] or previous['protocol']['selected_training']!=expected_ids
                or previous['artifact']['sha256']!=digest(args.warmup)):
            raise ValueError('the next lesson requires the passed predecessor artifact')
    # The unused 4e words stay visible in the report, never in the NP lesson.
    forms = set('a the black white small large cat dog bird horse runs sleeps chases sees is on near'.split())
    protocol = dict(seed=SEED,learning_rate=RATE,epochs=EPOCHS,batch_size=4,
        stage=args.stage,lesson='final whole-quarter gate, no updates' if final else 'one unknown construction, against the passed prerequisite',
        training_presentations=training_presentations,paired_updates=paired_updates,evaluation_presentations=len(evaluation),
        generated_corpus=dict(sentences=56,training=42,heldout=14),
        selected_training=[row['id'] for row in train],selected_heldout=[row['id'] for row in evaluation],
        restored_artifact=str(args.warmup),restored_sha256=digest(args.warmup),
        produced_artifact=str(args.warmup if final else args.out/'trained.ckpt'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        menu=MENU,modification_targets=('intersection','product'),family_order=[args.stage],sequence=STAGES,
        curriculum='one unknown at a time; passed predecessor required; successful path 84 presentations / 44 paired updates, plus the discarded branch 24 / 12; final whole-quarter gate follows the last lesson',
        within_family='seeded permutation; partial batch of three for each bare-NP lesson',
        rng=('all random streams reset to the declared seed after stage-3 restore' if previous is None else
             'model random states restored from the predecessor training artifact, before its held-out readings'),
        optimizer='fresh per lesson; the experiment artifacts contain model state and RNG, not optimizer moments',
        heldout='adjective distance 3 modulo 4; bare (noun+determiner) modulo 4 equals 3',
        score='sum of each carried word occurrence probability on the formal input arguments, coefficient 1; no phrase renormalization',
        unseen='uniform over all formal inputs and outputs; observed words use normalized counts without smoothing',
        word_flow='spec section 2b: the listed symmetric operations carry all operand words; product has an empty carrier; not/non preserve; headed operations carry the head operand words; clauses carry none; no first-word selector',
        category_evidence='kept readings only, every input and output; no annotation operands',
        questions='none for bare NPs; what is <adjective> ? otherwise, supplied answer only during training',
        heldout_labels='grader only; no model call receives a held-out tree, span, answer or occurrence label',
        noun_prior='table only; not read by chooser',
        reference_variants='unchanged',nested_marker_pattern=None,
        compose_step_cost='WHAT_STEP_COST for every binary, unary or transpose step',
        gates=dict(exact_brackets=.9,one_order1_occurrence=.95,duplicates=0,collisions=0,
                   unexpected_occurrences=0,exact_answers=1.),
        measurement='final stage: all 14 held-out brackets, 10 determiner NPs, 4 bare controls; otherwise the current lesson only; lowering stamps, kept operators and bare-description occurrences reported; absent families have no gate',
        deferred='the full NP→VP fixture requires all five lessons; stop the sequence at the first failure',
        budget='paired updates only; batch-end update disabled; stop at a failed gate',
        authorized_np_budget=dict(cumulative_presentations=108,cumulative_paired_updates=56,
            discarded_branch_presentations=24,discarded_branch_updates=12,
            reused_adjective_heldout=4,final_heldout=14),
        corpus_sha256={path.name:digest(path) for path in sorted(DEST.iterdir()) if path.is_file()})
    save(args.out/'protocol.json',protocol)
    save(args.out/'source-manifest.json',source_snapshot(Path(__file__).resolve().parents[1]))
    started=time.monotonic(); progress=[]; records=[]; results=[]
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model=make_model(args.out,patch)
        try:
            drift=restore_warmup(model,args.warmup,restore)
            if previous is None:
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
            parts=model.languageSpace.parts_of_speech
            if previous is None and parts.counts:
                raise ValueError('stage-3 artifact must start with empty occupation evidence')
            model.languageSpace.language_layer.operation_layer.set_stage_menu(MENU)
            identity=model.perceptualSpace.percept_store.identity
            if identity.nested or identity._marker_pattern is not None:
                raise ValueError('NP stage requires whole words with no marker lesson')
            save(args.out/'before.json',dict(loader_drift=drift,**tables(model,forms)))
            from GrammarLessons import _annotation_tree
            for lesson in lessons:
                _annotation_tree(model.languageSpace,lesson['tree'],len(lesson['text'].split()))
            data=model.inputSpace.data
            optimizer=model.getOptimizer(lr=RATE)
            if any(group['lr']!=RATE for group in optimizer.param_groups):
                raise ValueError('restored optimizer differs from the declared learning rate')
            trajectory=NPTrajectory(patch)
            kept_word_ids=[]
            native_observe=parts.observe_kept
            def occupation_audit(program,*rules):
                kept_word_ids.append([] if program is None or program.word_ids is None else program.word_ids.tolist())
                return native_observe(program,*rules)
            patch.setattr(parts,'observe_kept',occupation_audit)
            with paired_updates_only(model,optimizer) as updates:
                def present(rows,indices,*,training,epoch):
                    batch=[rows[index] for index in indices]
                    split='train' if training else 'validation'
                    q=[questions['train' if training else 'eval'][row['id']] for row in batch]
                    asked=all(value is not None for value in q)
                    if asked != any(value is not None for value in q):
                        raise ValueError('question and plain readings must have separate batches')
                    data.has_supervised_outputs=training and asked
                    data.grammar_lessons={'train':[lessons[i] for i in indices]} if training else {}
                    data.text_answers={'train':[value['answer'] for value in q]} if training and asked else {}
                    data.train_input=[row['text'] for row in batch] if training else []
                    data.train_output=[torch.zeros(1) for row in batch] if training and asked else []
                    data.source_addresses={split:[dict(document=row['id'],sentence=0,
                        document_key=row['id'],sentence_index=0,split=split) for row in batch]}
                    prompts=[value['prompt'] for value in q] if asked else None
                    factory=What.supervised if training else What.inference
                    requests=tuple(factory(b,split=split,prompt=prompts[b]) if asked else
                                   What.present(b,split=split) for b in range(len(batch)))
                    with observe(model) as captured, torch.set_grad_enabled(training), (
                            prompt_need(model,prompts) if asked else nullcontext()):
                        inputs=model.inputSpace.prepPackedInput([[row['text']] for row in batch])
                        model.runBatch(train=training,optimizer=optimizer if training else None,
                            batchNum=len(progress),batchSize=len(batch),split=split,
                            batch_override=(inputs,torch.zeros(len(batch),1) if training and asked else torch.empty(len(batch),0)),
                            source_rows=[[b] for b in range(len(batch))],questions=requests)
                    if len(kept_word_ids)!=len(batch):
                        raise RuntimeError('NP stage requires one native sentence transaction per row')
                    raw=model._last_sentence_field
                    if training:
                        if updates['paired'] != 2*(len(progress)+1):
                            raise RuntimeError('NP stage exceeded the paired-update budget')
                        trajectory.finish_batch(len(progress)+1,epoch,batch,captured,raw['wins'])
                        save(args.out/'trajectory-batches.json',trajectory.batches)
                        save(args.out/'trajectory.json',trajectory.report())
                    answers=list(getattr(getattr(model,'_last_answer_construction',None),'texts',()) or ())
                    for b,row in enumerate(batch):
                        observed=captured.get((0,b))
                        records.append(dict(id=row['id'],training=training,epoch=epoch,observed=observed,
                            word_ids=kept_word_ids[b],unidentified_word_leaves=sum(word<0 for word in kept_word_ids[b])))
                        if not training:
                            result=grade(labels[row['id']],observed)
                            result['question_expected']=q[b] is not None
                            if asked:
                                result['question']=q[b]['prompt']
                                result['answer']=answers[b] if b<len(answers) else None
                                result['answer_exact']=result['answer']==q[b]['answer']
                            results.append(result)
                    report=dict(batch=len(progress)+1,training=training,epoch=epoch,
                        ids=[row['id'] for row in batch],updates=dict(updates),elapsed_seconds=time.monotonic()-started)
                    progress.append(report)
                    kept_word_ids.clear()
                    save(args.out/'progress.json',progress);save(args.out/'kept-programs.json',records)
                    print(json.dumps(report),flush=True)
                    reset(model,len(batch))
                rng=np.random.default_rng(SEED)
                # This data-order RNG is separate from the model's archived
                # RNG. Advance it over the fixed earlier partitions, without
                # presenting or rescoring any example.
                for earlier in (() if final else STAGES[:STAGES.index(args.stage)]):
                    prior_rows=[row for row in all_training if lesson_stage(row)==earlier]
                    for _ in range(EPOCHS):
                        list(batches(prior_rows,rng))
                for epoch in range(EPOCHS):
                    for indices in batches(train,rng):
                        present(train,indices,training=True,epoch=epoch)
                assert updates['paired']==paired_updates and sum(len(row['ids']) for row in progress)==training_presentations
                save(args.out/'after.json',tables(model,forms))
                artifact_path=args.warmup if final else args.out/'trained.ckpt'
                if not final:
                    model.save_weights(str(artifact_path))
                artifact=dict(path=str(artifact_path),sha256=digest(artifact_path))
                parts.frozen=True
                model._concept_owner()._online_learning_frozen=True
                for indices in batches(evaluation):
                    present(evaluation,indices,training=False,epoch=None)
                assert updates['paired']==paired_updates
                summary=summarize_stage(results,args.stage)
                save(args.out/'results.json',dict(completed=True,protocol=protocol,**summary,
                    artifact=artifact,results=results,updates=dict(updates),
                    trajectory=trajectory.report() if trajectory.batches else dict(blocks=[],overall=None),
                    elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(completed=True,**summary)),flush=True)
        except BaseException:
            save(args.out/'failure.json',dict(error=traceback.format_exc(),completed_batches=len(progress),
                training_presentations=sum(len(row['ids']) for row in progress if row['training'])))
            raise
        finally:
            model.End()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--warmup',type=Path,required=True)
    parser.add_argument('--stage',choices=(*STAGES,'final'),default='lower')
    run(parser.parse_args())
