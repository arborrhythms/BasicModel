"""The predeclared native item-9 learning comparison. Measurements, not tests."""
import argparse
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]


@contextmanager
def eager_native_loops():
    """Use the prior native measurement's CPU dispatcher, with the same cells.

    This does not measure compiled throughput. In particular, changing
    vocabulary-bank shapes must not compile a new CPU graph on every batch.
    """
    import torch
    original = torch.while_loop
    def execute(condition, body, carries):
        while bool(condition(*carries)):
            carries = body(*carries)
        return carries
    torch.while_loop = execute
    try:
        yield
    finally:
        torch.while_loop = original


def config_for(seed, control, path):
    tree = ET.parse(ROOT / 'doc/benchmarks/2026-09-21-item2/packed.xml')
    root = tree.getroot()
    for parent, tag in (('architecture', 'conceptualWidth'), ('PartSpace', 'maxVectors')):
        element = root.find(parent + '/' + tag)
        if element is not None:
            root.find(parent).remove(element)
    changes = {'architecture/training/seed': seed,
               'architecture/data/maxDocs': 256,
               'architecture/training/reconstructionBasisLimit': 512,
               'PartSpace/nVectors': 32768, 'ConceptualSpace/nVectors': 65536,
               'ConceptualSpace/activeVectors': 32768}
    if control == 'reconstruction_only':
        changes.update({'architecture/training/interLossWeight': 0,
                        'architecture/training/expectationPolicyWeight': 0,
                        'architecture/training/expectationGain': 0})
    for key, value in changes.items():
        root.find(key).text = str(value)
    root.find('architecture/weightsPath').text = str(path.parent / 'unused.ckpt')
    tree.write(path, encoding='unicode')


class ContextControl:
    """Target-free corruption of native contexts, stable across loss replay."""
    def __init__(self, head, control, seed):
        self.forward = head.forward
        self.control = control
        self.seed = seed
        self.reset()

    def reset(self):
        self.bank = deque(maxlen=128)
        self.choices = OrderedDict()
        self.rng = random.Random(self.seed + 901)
        self.counts = dict(new_contexts=0, replayed=0, foreign=0, cold=0)
        self.enabled = True

    def detach(self):
        self.bank = deque(((key, x.detach(), m.detach()) for key, x, m in self.bank), maxlen=128)
        self.choices = OrderedDict((key, (x.detach(), m.detach()))
                                   for key, (x, m) in self.choices.items())

    def __call__(self, values, masks):
        import torch
        if not self.enabled or self.control in ('ordered', 'reconstruction_only'):
            return self.forward(values, masks)
        if self.control == 'context_free':
            return self.forward(torch.zeros_like(values), torch.zeros_like(masks))
        digest = hashlib.sha256(values.detach().cpu().contiguous().numpy().tobytes()
                                + masks.detach().cpu().contiguous().numpy().tobytes()).digest()
        if digest in self.choices:
            inputs = self.choices[digest]
            self.counts['replayed'] += 1
        else:
            eligible = [(x, m) for key, x, m in self.bank if key != digest and x.shape == values.shape]
            inputs = (self.rng.choice(eligible) if eligible else
                      (torch.zeros_like(values), torch.zeros_like(masks)))
            self.counts['new_contexts'] += 1
            self.counts['foreign' if eligible else 'cold'] += 1
            self.choices[digest] = inputs
            if len(self.choices) > 512:
                self.choices.popitem(last=False)
            self.bank.append((digest, values, masks))
        return self.forward(*inputs)


def discrimination(model):
    import torch
    from CategoricalDiscrimination import FIXED_PROBES, fixed_probe_discrimination
    from What import What
    from test_packed_reconstruction_parity import reset
    readings = {}
    # A serial seal is the reading. No parallel/order-0 field execution.
    for name, probe in FIXED_PROBES.items():
        roots = []
        for text in probe['texts']:
            raw = model.inputSpace.prepInput([text])
            with torch.no_grad():
                model.runBatch(train=False, split='runtime', batchSize=1,
                    batch_override=(raw, torch.empty(1, 0)), exploration_trial=True,
                    questions=(What.present(0, split='runtime'),))
            program, = model._last_understanding.answer_program
            roots.append(program.end_state.detach().cpu().flatten())
            reset(model, packed=False, final=True, batch=1)
        readings[name] = torch.stack(roots)
    return dict(scope='serial sealed-root codes; fixed probes, no gradient updates',
                **fixed_probe_discrimination(readings))


def thought_comparison(model, store_start, destination):
    import torch
    from Layers import ExpectationComparison, MeaningExpectation
    store = model.symbolSpace.ltm_store
    pairs, unsupported = [], 0
    for i in range(len(store)):
        if int(store.occurrence_id[i]) < store_start:
            continue
        provenance = store.expectation_of(i)
        if provenance is None or provenance['kind'] != 'observation':
            continue
        pair = store.expectation_pair(i)
        pairs.append(pair)
    checkpoint = destination / 'thought-start.ckpt'
    # Checkpoint the actual native model/memory. Both gains start here.
    for pair in pairs:
        model._selected_thought_chooser(replace(pair['observation'], mode='interrogative'))
    model.save_weights(checkpoint)
    initial_rng = torch.get_rng_state()
    trials = {}
    for gain in (0., 1.):
        assert model.load_weights(checkpoint, strict=True, require_match=True)
        torch.set_rng_state(initial_rng)
        model.eval()
        model.expectation_gain = gain
        samples = []
        for pair in pairs:
            meaning = pair['observation']
            question = replace(meaning, mode='interrogative')
            prediction = MeaningExpectation(pair['estimate'].roles, pair['presence_logits'])
            comparison = ExpectationComparison(prediction, meaning.roles, meaning.role_mask,
                pair['residual'], pair['presence_residual'], pair['document'],
                pair['source_occurrences'], pair['stream'])
            discourse = model.symbolSpace.discourse
            discourse.ensure_batch(1)
            discourse._last_expectation_comparisons[0] = comparison
            try:
                with model._query_boundary_scope((0,)), torch.no_grad():
                    result = model.run_selected_thought(question, row=0, work_budget=64)
                answer = [result.evidence['support_true'], result.evidence['support_false']]
                samples.append(dict(work=result.work.spent,
                    steps=sum(record.kind == 'thought' for record in result.records), answer=answer,
                    assertion_brier=(1-answer[0])**2 + answer[1]**2,
                    occupied_roles=meaning.role_mask.tolist()))
            except (ValueError, NotImplementedError) as error:
                unsupported += 1
                samples.append(dict(error=type(error).__name__, message=str(error)))
            finally:
                model._end_finished_selected_thought_episodes()
        trials[str(gain)] = samples
    return dict(pairs=len(pairs), unsupported=unsupported, trials=trials,
                target='positive assertion as presented; no world-truth certification')


def run(args):
    import numpy as np
    import torch
    import Language
    from Models import BaseModel
    from data import TheData
    from util import init_config, init_device, init_compile_backend
    from bench_sentence_expectation import (meaning_windows, _score_head, _controlled_inputs,
        observe_optimizer_steps, reconstruction_observations)
    from bounded_tests import source_snapshot
    torch.set_num_threads(1)
    for rng in (random.seed, np.random.seed, torch.manual_seed):
        rng(args.seed)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    config = destination / 'model.xml'
    config_for(args.seed, args.control, config)
    init_device('cpu')
    init_compile_backend('none')
    init_config(str(config), defaults_path=str(ROOT/'data/model.xml'))
    cfg = BaseModel.load_config(str(config))
    dat = dict(cfg['architecture']['data'])
    TheData.load('text', num_shards=1, max_docs=256, shard_dir=dat['shardDir'],
                 dat=dat, max_sentence_words=8)
    Language.TheGrammar._configured = False
    model, _ = BaseModel.from_config(str(config), data=TheData)
    model.set_sigma(0)
    model.checkpoint_every_batches = 0
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.reconstruction_placement = 'eager'
    assert model.serial and not model.mode_schedule.every
    discourse = model.symbolSpace.discourse
    head = discourse._inter_predictor
    control = ContextControl(head, args.control, args.seed)
    head.forward = control
    original_batch, original_observe, original_compact = model.runBatch, discourse._observe_meanings, model.post_tick_compact
    phase, captured, completed = None, {}, 0
    inverse_failures = Counter()
    reverse_methods = {}
    for arity, name in ((1, 'reverse_unary_step'), (2, 'reverse_binary_step')):
        original = getattr(model.languageSpace, name)
        reverse_methods[name] = original
        ops = list(model.languageSpace._tree_layer(arity).ops)
        names = [getattr(getattr(op, 'gl', op), 'rule_name', type(op).__name__) for op in ops]
        def observed_reverse(*a, _original=original, _names=names, **kw):
            result = _original(*a, **kw)
            if kw.get('return_status'):
                # The output walk supplies its own operator list. Local
                # indices there cannot be named using the compose catalogue.
                supplied = kw.get('ops')
                names = ([getattr(getattr(op, 'gl', op), 'rule_name', type(op).__name__)
                          for op in supplied] if supplied is not None else _names)
                scope = 'generation' if supplied is not None else 'input'
                for index in a[1].reshape(-1)[result[-1]].detach().cpu().tolist():
                    label = names[index] if 0 <= index < len(names) else 'invalid-rule'
                    inverse_failures[scope + ':' + label] += 1
            return result
        setattr(model.languageSpace, name, observed_reverse)
    report = dict(seed=args.seed, control=args.control, source=source_snapshot(ROOT),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        source_manifest=TheData.source_manifest, torch=torch.__version__, threads=1,
        device='cpu', updates=64, batch_size=2, backend='eager tensor loop', phases=[])
    def save():
        (destination/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    def observe(self, depths, payloads, tetralemmas, mask, layout, role_masks):
        value = original_observe(depths, payloads, tetralemmas, mask, layout, role_masks)
        if phase and phase['split'] == 'validation':
            for row, payload in enumerate(payloads):
                if payload is None or not int(depths[row]) or (mask is not None and not bool(mask[row])):
                    continue
                roles, occupied = self._canonical_meaning(payload, depths[row], layout,
                    None if role_masks is None else role_masks[row])
                if bool(occupied.any()):
                    captured.setdefault((row, self._expectation_documents[row]), []).append(
                        (roles.detach().cpu(), occupied.detach().cpu()))
        return value
    def stepped(*_):
        nonlocal completed
        completed += 1
    def batch(*a, **kw):
        before, started = completed, time.perf_counter()
        inverse_failures.clear()
        result = original_batch(*a, **kw)
        fidelity = reconstruction_observations(model, result[0])
        # This runtime flag also includes unsupported inverses and malformed
        # reverse programs. It is not a candidate-limit flag. Preserve such
        # rows in the complete run and report the actual candidate bound apart.
        sentence_ids = model.inputSpace._ar_concept_lookup_sentence_ids
        rows = model.inputSpace._ar_concept_lookup_rows
        per_sentence = [int(((ids == sid) & (r >= 0)).sum())
            for ids, r in zip(sentence_ids, rows) for sid in ids.unique() if int(sid) >= 0]
        fidelity['basis_limited_sentences'] = sum(n > model.reconstruction_basis_limit for n in per_sentence)
        fidelity['maximum_sentence_basis'] = max(per_sentence, default=0)
        fidelity['unavailable_inverse_calls'] = dict(inverse_failures)
        phase['steps'].append(dict(reconstruction=float(result[0].lossIn.detach()),
            fidelity=fidelity, optimizer_steps=completed-before,
            sources=copy.deepcopy(kw.get('source_rows')), seconds=time.perf_counter()-started))
        save()
        return result
    def compact(*a, **kw):
        result = original_compact(*a, **kw)
        control.detach()
        return result
    model.runBatch, model.post_tick_compact = batch, compact
    discourse._observe_meanings = types.MethodType(observe, discourse)
    optimizer = model.getOptimizer(lr=.0005)
    hook = observe_optimizer_steps(optimizer, stepped)
    initial_head = {name: p.detach().clone() for name, p in head.named_parameters()}
    fixed = None
    try:
        for name, split, opt, count in (('before', 'validation', None, 16),
                ('training', 'train', optimizer, 64), ('after', 'validation', None, 16)):
            phase = dict(name=name, split=split, steps=[])
            report['phases'].append(phase)
            captured.clear()
            control.reset()
            store_start = int(model.symbolSpace.ltm_store._next_occurrence)
            started = time.perf_counter()
            model.runEpoch(optimizer=opt, batchSize=2, split=split, max_batches=count)
            phase.update(seconds=time.perf_counter()-started, expectation=discourse.expectation_metrics(),
                         context_control=dict(control.counts))
            if len(phase['steps']) != count:
                raise RuntimeError(f'{name} exhausted: {len(phase["steps"])} of {count} batches')
            assert all(s['optimizer_steps'] == int(opt is not None) for s in phase['steps'])
            phase['reconstruction_mean'] = statistics.mean(s['reconstruction'] for s in phase['steps'])
            if captured:
                groups = list(captured.values())
                views = meaning_windows([torch.stack([v for v, _ in rows]) for rows in groups],
                    discourse._inter_chain_window, [torch.stack([m for _, m in rows]) for rows in groups])
                if views is None:
                    raise RuntimeError('held-out document split has no prediction pairs')
                if fixed is None:
                    fixed = views
                control.enabled = False
                phase['scores'] = {}
                for label in ('ordered', 'shuffled', 'context_free'):
                    x, m = _controlled_inputs(views[0], views[1], label, args.seed + 8042)
                    phase['scores'][label] = _score_head(head, x, m, views[2], target_masks=views[4])
                label = args.control if args.control != 'reconstruction_only' else 'ordered'
                x, m = _controlled_inputs(fixed[0], fixed[1], label, args.seed + 8042)
                phase['fixed_pretraining_encoding'] = _score_head(head, x, m, fixed[2], target_masks=fixed[4])
                control.enabled = True
            save()
        assert completed == 64
        report['predictor_delta'] = sum(float((p.detach()-initial_head[name]).square().sum())
            for name, p in head.named_parameters()) ** .5
        # Measurement-only forwards no longer belong to the recorded native epochs.
        model.runBatch, model.post_tick_compact = original_batch, original_compact
        discourse._observe_meanings = original_observe
        control.enabled = False
        if args.control == 'ordered':
            report['thought'] = thought_comparison(model, store_start, destination)
            save()
        report['discrimination'] = discrimination(model)
        report['source_unchanged'] = report['source'] == source_snapshot(ROOT)
        assert report['source_unchanged']
        save()
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        hook.remove()
        model.runBatch, model.post_tick_compact = original_batch, original_compact
        discourse._observe_meanings = original_observe
        for name, original in reverse_methods.items():
            setattr(model.languageSpace, name, original)
        head.forward = control.forward
        model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--control', choices=('ordered', 'shuffled', 'context_free', 'reconstruction_only'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    with eager_native_loops():
        run(parser.parse_args())
