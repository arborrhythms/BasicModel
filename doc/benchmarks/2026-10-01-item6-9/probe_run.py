"""One fresh 400-epoch variant run, with passive derivation measurements."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('variant', choices=('a', 'b', 'c'))
parser.add_argument('output', type=Path)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(exist_ok=True)
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
os.environ.pop('BASIC_SEED', None)
os.environ.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false')
warnings.filterwarnings('ignore')
import torch
import Models
import Language
import Spaces
import SentenceCompose
import probe_variants
from probe_observation import full_choice

probe_variants.install(args.variant, Models, SentenceCompose)

# Execute only the pure mathematical definitions from the preserved Claude
# probe. Its original training entry point and hooks are not executed here.
import itertools
TARGET = ['hello world', 'hello there', 'loving world', 'loving there']
Y = torch.tensor([-1., 1., 1., -1.], dtype=torch.float64)
BIN = {'conjunction': torch.minimum, 'disjunction': torch.maximum,
       'intersection': torch.minimum, 'union': torch.maximum}
original = ast.parse((HERE / 'deriv69.py').read_text())
for node in original.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('margin', 'simulate', 'enumerate_full', 'align'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(HERE / 'deriv69.py'), 'exec'))

STATE = dict(model=None, items=None, batch=-1, epoch=-1, calls=[], records=[],
             first=None, last=None, evaluation=None, gradients=[], pairs=0,
             train_pairs=0, optimizer_steps=0, max_rebuild_deviation=0.,
             max_root_deviation=0., first_training_items=None)
trial_log = (out / 'trials.jsonl').open('w')
pair_log = (out / 'pairs.jsonl').open('w')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metric(roots):
    value = margin(roots)
    return dict(exact=value['exact'], max_weight=value['max'],
                l2_weight=value['norm'], sv3=value['sv3'])


def available(leaves):
    choices = enumerate_full(leaves)
    ranked = sorted(((metric(roots), expression) for expression, roots in choices.items()),
                    key=lambda pair: (not pair[0]['exact'], pair[0]['max_weight']))
    best, expression = ranked[0]
    return dict(derivation=expression, margin=best, expressions=len(choices),
        readable_at_5=sum(m['exact'] and m['max_weight'] <= 5 for m, _ in ranked),
        leaves=[[v.tolist() for v in row] for row in leaves])


prep_original = Spaces.InputSpace.prepInput
def prep(self, items, *a, **k):
    try:
        STATE['items'] = [str(x).replace('\x00', '').strip() for x in items]
    except Exception:
        STATE['items'] = None
    return prep_original(self, items, *a, **k)
Spaces.InputSpace.prepInput = prep

run_original = Models.BasicModel.runBatch
def run(self, *a, **k):
    STATE['model'] = self
    STATE['batch'] += 1
    if k.get('train', a[0] if a else False):
        STATE['epoch'] += 1
    return run_original(self, *a, **k)
Models.BasicModel.runBatch = run

selection_original = Language.LanguageSpace.choose_operation
def selection(self, state, row_gate, **kw):
    result = selection_original(self, state, row_gate, **kw)
    choice = result[0] if isinstance(result, tuple) else result
    layer = self.language_layer.operation_layer
    STATE['calls'].append(full_choice(state, choice,
        (list(layer.op_names or []), list(layer.unary_names or []))))
    return result
Language.LanguageSpace.choose_operation = selection


def observe_trial(path, trial, cost, training):
    order = align(STATE['items'])
    if order is None or not STATE['calls']:
        return None
    stacks, leaves, deviation = simulate(STATE['calls'])
    stacks, leaves = [stacks[i] for i in order], [leaves[i] for i in order]
    record = dict(batch=STATE['batch'], epoch=STATE['epoch'], training=training,
        trial=trial, derivations=[[expression for expression, _ in row] for row in stacks],
        depths=[len(row) for row in stacks], cost=cost.detach().double()[order].tolist(),
        rebuild_deviation=max(deviation[i] for i in order), margin=None)
    if all(len(row) == 1 for row in stacks):
        roots = torch.stack([row[0][1] for row in stacks])
        record['margin'] = metric(roots)
        sid = STATE['model']._open_sentence_slot
        actual = path[1][9][:, sid].detach().double()[order]
        record['root_deviation'] = float((roots - actual[:, :roots.shape[-1]]).abs().max())
        STATE['max_root_deviation'] = max(STATE['max_root_deviation'], record['root_deviation'])
        assert record['root_deviation'] < 2e-5, record
    STATE['max_rebuild_deviation'] = max(STATE['max_rebuild_deviation'], record['rebuild_deviation'])
    assert record['rebuild_deviation'] < 2e-5, record
    assert all(len(row) == 3 for row in leaves), [len(row) for row in leaves]
    # Keep every operation choice but do not retain all the large tensor slabs.
    record['operations'] = [{key: value for key, value in call.items() if key != 'x'}
                            for call in STATE['calls']]
    trial_log.write(json.dumps(record, allow_nan=False) + '\n')
    trial_log.flush()
    if trial == 0:
        if training:
            if STATE['first'] is None:
                STATE['first'] = (STATE['epoch'], leaves)
                STATE['first_training_items'] = list(STATE['items'])
            STATE['last'] = (STATE['epoch'], leaves)
        else:
            STATE['evaluation'] = dict(record=record, leaves=leaves, items=list(STATE['items']))
    return record


pair_original = SentenceCompose.sentence_pair
def pair(cache, compose, score, step, *, active, training=True):
    model = STATE['model']
    expected = (5 if args.variant == 'b' else 2) if training else 1
    versions, scored, steps = [], [], 0
    def compose_observed(value, prior):
        STATE['calls'] = []
        versions.append(tuple(p._version for p in model.parameters()))
        assert not steps, 'a trial was composed after training began'
        return compose(value, prior)
    def score_observed(path, alternative):
        result = score(path, alternative)
        assert versions[-1] == tuple(p._version for p in model.parameters())
        record = observe_trial(path, len(scored), result[0], training)
        component = getattr(model, '_probe_answer_component', None)
        if training and record is not None:
            assert torch.is_tensor(component)
            assert component.requires_grad == (args.variant == 'c')
            record['answer_error_has_gradient'] = component.requires_grad
            if args.variant == 'c' and STATE['epoch'] == 0:
                params = tuple(model.parameters())
                gradients = model._sentence_pullback.gradients(component.mean(), params)
                names = {id(p): n for n, p in model.named_parameters()}
                nonzero = {names.get(id(p), str(i)): float(g.detach().norm())
                           for i, (p, g) in enumerate(zip(params, gradients))
                           if g is not None and bool(g.detach().ne(0).any())}
                STATE['gradients'].append(dict(trial=len(scored), norms=nonzero))
        scored.append(record)
        return result
    def step_observed(loss):
        nonlocal steps
        assert len(scored) == expected, 'not all trials were costed before the first update'
        assert all(version == versions[0] for version in versions)
        steps += 1
        return step(loss)
    result = pair_original(cache, compose_observed, score_observed, step_observed,
                           active=active, training=training)
    assert len(scored) == expected and steps == (expected if training else 0)
    STATE['pairs'] += 1
    STATE['train_pairs'] += int(training)
    STATE['optimizer_steps'] += steps
    indexes = result[2].long()
    expected_winner = torch.where(active, result[1].argmin(-1), torch.zeros_like(indexes))
    torch.testing.assert_close(indexes, expected_winner, rtol=0, atol=0)
    pair_log.write(json.dumps(dict(batch=STATE['batch'], epoch=STATE['epoch'], training=training,
        items=STATE['items'], winners=indexes.tolist(), costs=result[1].tolist(),
        all_costed_before_training=True, equal_parameter_versions=True, optimizer_steps=steps)) + '\n')
    pair_log.flush()
    return result
SentenceCompose.sentence_pair = pair

started = time.monotonic()
try:
    model = Models.ModelFactory.run('data/XOR_grammar.xml')[0][2]
    data = model.inputSpace.data
    answers = torch.stack(data.reconstructed_output).reshape(-1).detach().double()
    targets = torch.stack(data.test_output).reshape(-1).detach().double()
    assert answers.numel() == targets.numel() == 4 and torch.isfinite(answers).all()
    correct = int(((answers > .5) == (targets > .5)).sum())
    error = float((answers - targets).square().mean())
    final = STATE['evaluation']
    assert final is not None and STATE['first'] is not None and STATE['last'] is not None
    assert STATE['train_pairs'] == 400, STATE['train_pairs']
    report = dict(variant=args.variant, seconds=time.monotonic()-started,
        answers=answers.tolist(), targets=targets.tolist(), test_inputs=[str(x) for x in data.test_input],
        mse=error, correct=correct, settled_bar=correct == 4 and error < .05,
        final_derivations=dict(zip(TARGET, final['record']['derivations'])),
        final_margin=final['record']['margin'],
        best_first_epoch=dict(epoch=STATE['first'][0], **available(STATE['first'][1])),
        best_last_epoch=dict(epoch=STATE['last'][0], **available(STATE['last'][1])),
        best_after_training=available(final['leaves']),
        training_pairs=STATE['train_pairs'], trial_optimizer_steps=STATE['optimizer_steps'],
        max_rebuild_deviation=STATE['max_rebuild_deviation'], max_root_deviation=STATE['max_root_deviation'],
        answer_gradient_samples=STATE['gradients'],
        patch_sha256=hashlib.sha256((HERE/'probe-patches'/f'{args.variant}.patch').read_bytes()).hexdigest(),
        candidate_source_unmodified=True)
    write(out / 'measurement.json', report)
    print('PROBE_RESULT ' + json.dumps({k:report[k] for k in
        ('variant','answers','mse','correct','settled_bar','final_derivations','final_margin')}), flush=True)
finally:
    trial_log.close()
    pair_log.close()
