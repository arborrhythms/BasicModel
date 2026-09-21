"""Held-out text measurement after the existing grammar wording curriculum.

Run from basicmodel: PYTHONPATH=bin:test .venv/bin/python <this file>.
The grammar is trained by the item-1 compose/generate recipe, then frozen.
This is a representation-controlled study, not joint production learning.
No annotated operator or answer enters prediction or controller execution.
"""
import os
os.environ.setdefault('MODEL_COMPILE', 'eager')
os.environ.setdefault('BASICMODEL_DEVICE', 'cpu')
import copy, hashlib, json, random, sys, tempfile, time
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace
from Layers import SentenceExpectation, ExpectationComparison, MeaningExpectation, TernaryTruthStore
from Models import _append_observed_meaning
from Meaning import negative_image
from Queries import QueryWorkBudget
from bounded_tests import source_snapshot
import test_compiled_word_chunk as fixtures
from test_output_walk import _capture_program_probe
from test_surface_grammar import test_real_text_has_a_complete_selected_meaning
from bench_sentence_expectation import _score_head, _controlled_inputs

OUT = Path('doc/benchmarks/2026-09-21-item2')
torch.set_num_threads(1)
report = {'source': source_snapshot(Path.cwd()), 'grammar_seed': 931,
          'encoder_training': 'item-1 supervised grammar curriculum, then frozen',
          'limitation': 'Paraphrase continuation study; not natural document prediction or joint utility.',
          'torch': torch.__version__, 'device': 'cpu', 'threads': 1, 'runs': []}
started = time.perf_counter()
with tempfile.TemporaryDirectory() as td, pytest.MonkeyPatch.context() as mp:
    original = fixtures._tiny_canonical_model
    owners = []
    def keep(*args, **kwargs):
        model = original(*args, **kwargs)
        owners.append(model)
        return model
    mp.setattr(fixtures, '_tiny_canonical_model', keep)
    test_real_text_has_a_complete_selected_meaning(Path(td), mp)
    model = owners[0]
    report['grammar_seconds'] = time.perf_counter() - started
    curriculum = json.loads(Path('data/grammar_wording.json').read_text())
    def capture(rows):
        found = []
        with torch.no_grad():
            for start in range(0, len(rows), 64):
                model._tensor_final_end_slots = None
                model._tensor_sentence_roots_live = None
                result = _capture_program_probe(model, [r['text'] for r in rows[start:start+64]])
                for row, program in zip(rows[start:start+64], result.answer_program):
                    meaning = model.languageSpace.program_meaning(program, model.grammatical_thoughts)
                    if meaning is not None:
                        found.append((row, meaning.detached()))
        return found
    train = capture(curriculum['train'])
    heldout = capture(curriculum['validation'] + curriculum['test'])
    # Real worded continuations share meaning; no synthetic programs substitute
    # for forward composition. Holdout preserves item 1's complete-wording split.
    window = model.symbolSpace.discourse._inter_chain_window
    def pairs(rows):
        groups = {}
        for row, meaning in rows:
            groups.setdefault(meaning.role_refs, []).append((row, meaning))
        x, masks, y, ym, descriptions = [], [], [], [], []
        for group in groups.values():
            if len(group) < 2:
                continue
            for i, (row, meaning) in enumerate(group):
                prior = group[(i-1) % len(group)][1]
                context = torch.zeros(window, *prior.roles.shape)
                context[-1] = prior.roles
                present = torch.zeros(window, 3, dtype=torch.bool)
                present[-1] = prior.role_mask
                x.append(context); masks.append(present); y.append(meaning.roles)
                ym.append(meaning.role_mask); descriptions.append((row, meaning, prior))
        return torch.stack(x), torch.stack(masks), torch.stack(y), torch.stack(ym), descriptions
    tx, tm, ty, tym, _ = pairs(train)
    vx, vm, vy, vym, descriptions = pairs(heldout)
    report.update(train_pairs=len(tx), heldout_pairs=len(vx), concept_width=ty.shape[-1])
    # Additional genuinely worded continuations change one noun or both nouns.
    semantic_rows, triples = [], []
    for i, (row, meaning, prior) in enumerate(descriptions):
        words = row['text'].split()
        operands = row.get('operands', ())
        if len(operands) != 2 or row['form'] not in ('part', 'whole', 'equal'):
            continue
        a, b = operands
        related = list(words); related[b] = 'wheels' if words[b] != 'wheels' else 'pages'
        unrelated = list(related); unrelated[a] = 'books' if words[a] != 'books' else 'bicycles'
        triples.append(i)
        semantic_rows.extend((dict(text=' '.join(related)), dict(text=' '.join(unrelated))))
    semantic = capture(semantic_rows)
    report['semantic_parse_coverage'] = [len(semantic), len(semantic_rows)]
    for seed in (0, 1, 2):
        runs, heads = {}, {}
        for control in ('ordered', 'shuffled', 'context_free'):
            torch.manual_seed(seed)
            head = SentenceExpectation(ty.shape[-1], window)
            x, mask = _controlled_inputs(tx, tm, control, seed+1000)
            xx, mm = _controlled_inputs(vx, vm, control, seed+2000)
            optimizer = torch.optim.Adam(head.parameters(), lr=.003)
            before = _score_head(head, xx, mm, vy, target_masks=vym)
            for step in range(1000):
                indices = torch.randint(len(x), (64,))
                optimizer.zero_grad(set_to_none=True)
                roles, logits = head(x[indices], mask[indices])
                loss = (roles - ty[indices]).square().mean() + F.binary_cross_entropy_with_logits(logits, tym[indices].float())
                loss.backward(); optimizer.step()
            runs[control] = dict(before=before, after=_score_head(head, xx, mm, vy, target_masks=vym), updates=1000)
            heads[control] = head
        head = heads['ordered']
        with torch.no_grad():
            predicted, logits = head(vx, vm)
        semantic_norms = []
        if len(semantic) == len(semantic_rows):
            for j, index in enumerate(triples):
                c1, _ = negative_image(semantic[2*j][1].roles, predicted[index], logits[index].sigmoid())
                c2, _ = negative_image(semantic[2*j+1][1].roles, predicted[index], logits[index].sigmoid())
                semantic_norms.append((float(c1.norm()), float(c2.norm())))
        # The same controller learns from unlabelled frozen parsed sequences.
        # Encoder and predictor are held fixed here to isolate policy credit.
        # This does not substitute for the native joint-learning measurement.
        model.selected_thought_choosers = torch.nn.ModuleDict()
        chooser = model._selected_thought_chooser(replace(descriptions[0][1], mode='interrogative'))
        initial_policy = copy.deepcopy(chooser.state_dict())
        disc = model.symbolSpace.discourse
        disc._inter_predictor.load_state_dict(head.state_dict())
        policy_reports = {}
        # Same ordinary chooser and same completed parsed questions. Each trial
        # starts from its own ordinary episode; no answer label enters it.
        trials = {}
        disc = model.symbolSpace.discourse
        model.selected_thought_policy_weight = 0.
        for gain in (0., 1.):
            torch.manual_seed(seed + 3000)
            chooser.load_state_dict(initial_policy)
            optimizer = torch.optim.Adam(chooser.parameters(), lr=.003)
            model.__dict__.pop('_expectation_policy_baseline', None)
            model.expectation_gain = gain
            model.expectation_policy_weight = .2
            model.expectation_query_budget = 64
            model.ltm_consolidation = True
            store = TernaryTruthStore(ty.shape[-1], capacity=4096)
            model.symbolSpace.ltm_store = store
            disc._ltm_store = store
            disc.Reset(); disc.ensure_batch(1); disc.train()
            disc.set_inter_loss_weight(0.)
            credits = []
            for step in range(120):
                model._stage_expectation_queries(training=True)
                meaning = train[(step // 4) % len(train)][1]
                disc.predict_and_observe_stm_end_state([int(meaning.role_mask.sum())], [meaning.roles],
                    layout='infix', role_masks=[meaning.role_mask])
                comparison = disc.last_expectation_comparison()
                index = _append_observed_meaning(store, meaning.roles, 3, meaning=meaning,
                    expectation=comparison, stream=0)
                disc.bind_observation_occurrence(0, store.occurrence_of(index))
                loss = model._expectation_policy_loss()
                if loss is not None:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                    credits.append(dict(model._expectation_policy_report))
                disc.detach_prediction_context()
            policy_reports[str(gain)] = credits
            model.eval()
            samples = []
            for i, (_row, meaning, _prior) in enumerate(descriptions):
                question = replace(meaning, mode='interrogative')
                disc._last_expectation_comparisons[0] = ExpectationComparison(
                    MeaningExpectation(predicted[i], logits[i]), meaning.roles, meaning.role_mask,
                    meaning.roles-predicted[i], meaning.role_mask.float()-logits[i].sigmoid(), None)
                with model._query_boundary_scope((0,)), torch.no_grad():
                    result = model.run_selected_thought(question, row=0, work_budget=64)
                model._end_finished_selected_thought_episodes()
                samples.append({'work': result.work.spent,
                    'steps': sum(r.kind == 'thought' for r in result.records),
                    'answer': [result.evidence['support_true'], result.evidence['support_false']],
                    'assertion_brier': (1-result.evidence['support_true'])**2 + result.evidence['support_false']**2})
            trials[str(gain)] = samples
        report['runs'].append({'seed': seed, 'predictor': runs, 'semantic_norms': semantic_norms,
            'chooser': '120 frozen-encoding unlabelled presentations; independent residual EMA',
            'policy_training': policy_reports,
            'gain_trials': trials, 'answer_disagreement': sum(a['answer'] != b['answer'] for a,b in zip(trials['0.0'], trials['1.0']))})
        (OUT/'text.json').write_text(json.dumps(report, indent=2)+'\n')
        print('TEXT SEED', seed, {k:v['after']['feature_mse'] for k,v in runs.items()}, flush=True)
report['seconds'] = time.perf_counter()-started
(OUT/'text.json').write_text(json.dumps(report, indent=2)+'\n')
