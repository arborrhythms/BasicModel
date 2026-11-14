"""Saved-model operator readings, with strict restoration in isolated processes."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore_model(config, checkpoint, *, config_sha256=None):
    import torch
    from eval_nanochat_grammar import build_eval_model
    from IntervalLog import validate_probe_copy
    if config_sha256 is not None and digest(config) != config_sha256:
        raise ValueError('recorded model configuration has changed')
    model, data = build_eval_model(config, autoload=False)
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        data.source_manifest = saved['training_state']['data_manifest']
        if not model.load_weights(str(checkpoint), strict=True, require_match=True):
            raise ValueError('saved stage model could not be restored')
        validate_probe_copy(model, saved)
        model.eval()
        model._tensor_peer_while_eager = True
        return model, data
    except BaseException:
        model.End()
        raise


def operator_parameters(model):
    """Read existing layer instances; never construct an operator to measure it."""
    from Language import TheGrammar
    from PartsOfSpeech import operation_name
    rules = tuple(TheGrammar.rules_upward)
    owners = {}
    for arity, label in ((1, 'unary'), (2, 'binary')):
        layer = model.languageSpace._tree_layer(arity)
        ops = () if layer is None else (layer.unary_ops if arity == 1 else layer.ops)
        for rule, wrapped in zip(getattr(model.languageSpace, f'_compose_{label}_rules'), ops):
            owners.setdefault(operation_name(rule), []).append(getattr(wrapped, 'gl', wrapped))
    hosts = model.symbolSpace.subspace._host_layer_registry
    names = {id(value): name for name, value in model.named_parameters()}
    result = {}
    for rule in rules:
        name = operation_name(rule)
        if not name or name in result:
            continue
        layers = owners.get(name, [])
        if not layers:
            layers = [layer for (role, method), layer in hosts.items() if method == rule.method_name]
        parameters = {}
        for layer in layers:
            for value in layer.parameters():
                if id(value) not in names:
                    raise ValueError('operator parameter has no model owner: ' + name)
                parameters[names[id(value)]] = value.detach().cpu().clone()
        result[name] = parameters
    return result


def parameter_change(current, baseline):
    import torch
    if current.keys() != baseline.keys():
        raise ValueError('operator parameter ownership changed between saved models')
    if not current:
        return dict(parameters=0, baseline_norm=None, delta_norm=None, relative_delta=None)
    squared_before = squared_change = 0.
    for name in current:
        a, b = current[name], baseline[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('operator parameter layout changed: ' + name)
        squared_before += float(b.double().square().sum())
        squared_change += float((a.double()-b.double()).square().sum())
    before, change = squared_before**.5, squared_change**.5
    return dict(parameters=sum(v.numel() for v in current.values()), baseline_norm=before,
                delta_norm=change, relative_delta=None if before == 0 else change/before)


def count_program(program, language):
    from PartsOfSpeech import operation_name
    if program is None or not len(program.leaves):
        raise ValueError('missing completed program is not a zero-use reading')
    counts, leaves, padded = Counter(), [], False
    for kind, local, position in program.actions.detach().cpu().tolist():
        if kind < 0:
            padded = True
            continue
        if padded:
            raise ValueError('non-padding action after completed program')
        if kind == 0:
            leaves.append(position)
            continue
        if kind not in (1, 2, 3):
            raise ValueError('unknown completed action')
        rules = getattr(language, '_compose_unary_rules' if kind == 2 else '_compose_binary_rules')
        if not 0 <= local < len(rules):
            raise ValueError('completed action has no owned rule')
        counts[operation_name(rules[local])] += 1
    if sorted(leaves) != list(range(len(program.leaves))):
        raise ValueError('completed program does not cover every leaf once')
    return counts


@contextmanager
def observe_kept_programs(model):
    """Read only winning, active programs at the existing commit boundary."""
    absent = object()
    local = model.__dict__.get('_commit_sentence', absent)
    original = model._commit_sentence
    reading = dict(counts=Counter(), sentences=0)
    def observed(state, sid, active, observations, predictions, wins):
        # Native sentence references expire during commit. Read the selected
        # journal while pinned, then publish counts only if commit succeeds.
        counts, sentences = Counter(), 0
        for batch in range(len(active)):
            if bool(active[batch]):
                program = observations[int(wins[batch])]['entries'][batch]
                counts.update(count_program(program, model.languageSpace))
                sentences += 1
        result = original(state, sid, active, observations, predictions, wins)
        reading['counts'].update(counts)
        reading['sentences'] += sentences
        return result
    model._commit_sentence = observed
    try:
        yield reading
    finally:
        if local is absent:
            del model._commit_sentence
        else:
            model._commit_sentence = local


def evaluate(config, checkpoint, manifest, output, *, lesson=None):
    """The caller owns this process; all held-out state and RNG draws stay here."""
    import torch
    import util
    from eval_nanochat_grammar import frozen_online_learning, _reset_model
    from corpus_lessons import heldout
    from eval_lessons import lesson_readings, criterion_reading
    torch.set_num_threads(1)
    util.init_device('cpu')
    util.TheCompileBackend = 'none'
    spec = json.loads(Path(manifest).read_text())
    model, data = restore_model(config, checkpoint)
    try:
        parameters = operator_parameters(model)
        torch.save(parameters, str(output)+'.parameters.pt')
        validation = json.loads((ROOT/spec['validation']).read_text())
        seen = []
        with frozen_online_learning(model), observe_kept_programs(model) as work, torch.no_grad():
            for item in (() if lesson else validation['items']):
                _reset_model(model)
                for index, text in enumerate(item['sentences']):
                    before = work['sentences']
                    with data.runtime_batch([text], document_keys=[('operator-validation',item['kind'],item['key'])]):
                        model.understand(model.inputSpace.prepInput([text]))
                    if work['sentences'] == before:
                        raise ValueError('validation text has no completed reading: ' + item['key'])
                    seen.append([item['kind'], item['key'], index])
                    Path(output).write_text(json.dumps(dict(status='incomplete',
                        validation_sentences=seen, partial_use=dict(work['counts'])),indent=2)+'\n')
        counts = work['counts']
        # Separate restores make each lesson independent of validation state.
        lessons = json.loads((ROOT/spec['lessons']).read_text())
        if lesson is None or lesson.startswith('math-'):
            from stage_models import math_stages
            maths = [dict(stage, criterion=spec['math_criterion'])
                     for stage in math_stages(ROOT, spec['training_documents'])]
            lessons = dict(stages=lessons['stages']+maths if lesson is None else maths)
        criteria = {}
        for stage in lessons['stages']:
            if lesson is not None and stage['name'] != lesson:
                continue
            model.End()
            model, data = restore_model(config, checkpoint)
            held = sorted(stage['heldout'], key=lambda d:d['document'])
            selected = dict(stage, heldout=held[:spec['heldout_documents']] if lesson else held)
            readings = lesson_readings(model, [])
            readings['answers'] = heldout(model, selected, data)
            criteria[stage['name']] = dict(readings=readings,
                criterion=criterion_reading(stage['criterion'], readings,
                    operations=spec['operations'].get(stage['name'], ())))
        if lesson is not None and lesson not in criteria:
            raise ValueError('unknown requested lesson: ' + lesson)
        if lesson is None:
            math_answers=[answer for name,value in criteria.items() if name.startswith('math-')
                          for answer in value['readings']['answers']]
            math_reading=dict(readings,answers=math_answers)
            criteria['math']=dict(readings=math_reading,
                criterion=criterion_reading(spec['math_criterion'],math_reading))
        probes = {}
        if lesson is None:
            from eval_games import game_rows, production_probe
            for name, rows in game_rows(ROOT/'data/games').items():
                model.End()
                model, data = restore_model(config, checkpoint)
                probes[name] = production_probe(model, rows)
        return dict(status='complete', checkpoint_sha256=digest(checkpoint),
            validation_sha256=digest(ROOT/spec['validation']), validation_sentences=seen,
            use=dict(counts), criteria=criteria,
            probes={name: probes.get(name, dict(status='unavailable', value=None,
                reason='no game for this rule')) for name in parameters})
    finally:
        model.End()


def operator_table(models, manifest):
    """Evaluate named checkpoint/config pairs without touching a live learner."""
    import torch
    import os
    sys.path.insert(0, str(ROOT/'test'))
    from bounded_tests import run_guarded
    if 'M0' not in models:
        raise ValueError('operator table requires the untrained M0 baseline')
    readings, parameters = {}, {}
    with tempfile.TemporaryDirectory(prefix='basicmodel-operators-') as temporary:
        for label, entry in models.items():
            if entry['checkpoint'] is None:
                readings[label] = dict(status='unavailable', reason='checkpoint was not produced')
                continue
            folder = Path(temporary)/str(len(readings)); folder.mkdir()
            output = folder/'reading.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--config', str(entry['config']),
                '--checkpoint', str(entry['checkpoint']), '--manifest', str(manifest), '--output', str(output)]
            result = run_guarded(command,cwd=ROOT,env=dict(os.environ),log_path=folder/'process.log',
                memory_bytes=8*1024**3,timeout=1800)
            if result['exit_code']:
                readings[label] = dict(status='unavailable', reason='evaluation process failed',process=result,
                    detail=(folder/'process.log').read_text()[-12000:])
                if output.exists():
                    readings[label]['partial'] = json.loads(output.read_text())
                if Path(str(output)+'.parameters.pt').exists():
                    parameters[label] = torch.load(str(output)+'.parameters.pt', map_location='cpu', weights_only=True)
                continue
            readings[label] = json.loads(output.read_text())
            parameters[label] = torch.load(str(output)+'.parameters.pt', map_location='cpu', weights_only=True)
    baseline = parameters.get('M0')
    rows = []
    if baseline is not None:
        for label, value in parameters.items():
            if value.keys() != baseline.keys():
                raise ValueError('saved operator catalogs differ')
            if (readings[label]['status']=='complete' and readings['M0']['status']=='complete'
                and readings[label]['validation_sentences'] != readings['M0']['validation_sentences']):
                raise ValueError('operator table did not read the same ordered validation set')
        for name in baseline:
            rows.append(dict(rule=name, models={label: dict(
                status=value['status'], n=value['use'].get(name, 0) if value['status']=='complete' else None,
                change=parameter_change(parameters[label][name], baseline[name]) if label in parameters else None,
                probe=value['probes'][name] if value['status']=='complete' else
                    dict(status='unavailable',value=None,reason='saved-model evaluation failed'))
                for label, value in readings.items()}))
    return dict(schema=1, readings=readings, rows=rows, learned_utility='unproven')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--lesson')
    args = parser.parse_args()
    result = evaluate(args.config, args.checkpoint, args.manifest, args.output, lesson=args.lesson)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
