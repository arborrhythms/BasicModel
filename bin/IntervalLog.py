"""Interval observations; probe execution belongs to a separate evaluation process."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def scalar(value):
    return None if value is None else float(value.detach().mean().cpu())


def validate_probe_copy(model, saved):
    """A probe must describe the checkpoint's tensors, byte for byte."""
    import torch
    actual = model.state_dict()
    expected = saved['state_dict']
    different = [name for name in sorted(actual.keys() | expected.keys())
                 if name not in actual or name not in expected
                 or actual[name].dtype != expected[name].dtype
                 or actual[name].shape != expected[name].shape
                 or not torch.equal(actual[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                    expected[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8))]
    if different:
        raise ValueError('Probe checkpoint restore changed tensors: ' + ', '.join(different))


def provisioned_luminosity(model):
    """Fold only provisioned fact poles with TruthLayer's code-space formula."""
    import torch
    store = model.symbolSpace.ltm_store
    truth = getattr(model.symbolSpace, 'truth_layer', None)
    if truth is None:
        return None
    positive = torch.zeros(truth.nDim, device=truth.truths.device)
    negative = torch.zeros_like(positive)
    with torch.no_grad():
        for row in store.rows_of_origin(store.ORIGIN_PROVISIONED).tolist():
            if store.kind_of(row) != 'fact':
                continue
            code = truth._flatten_ltm_row(store, row).to(positive)
            for strength in (store.truth_plus[row], -store.truth_minus[row]):
                value = code * strength
                positive = torch.maximum(positive, value.clamp_min(0))
                negative = torch.maximum(negative, (-value).clamp_min(0))
        return float(((positive-negative)-torch.minimum(positive, negative)).mean().clamp(-1, 1))


def captured_metrics(model):
    store = model.symbolSpace.ltm_store
    costs = model.errors.breakdown()
    owner = model._concept_owner()
    rows = set(owner._csw_rows.values())
    inventory = sum(owner._order0_inventory_row(row) for row in rows)
    return dict(reconstruction_loss=scalar(model._last_primary_costs.get('input_reconstruction')),
        expectation_discrepancy={name: item['value'] for name, item in costs.items()
                                if item['category'] == 'expectation'},
        ltm=dict(occupied=len(store), capacity=store.capacity, forgetting_passes=int(store.forget_passes),
                 last_forgetting_report=getattr(store, 'last_forgetting_report', None)),
        provisioned_truth_luminosity=provisioned_luminosity(model),
        order_zero_inventory=dict(rows=int(inventory), nVectors=int(owner.nVectors)))


def append_report(report):
    from util import TheXMLConfig
    path = Path(TheXMLConfig.training('intervalLogFile', None) or ROOT/'output/intervals.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(report, allow_nan=False, sort_keys=True)+'\n')


def observe_forgetting(model, report):
    """Capture each completed forgetting pass, before another can replace it."""
    from util import TheXMLConfig
    if not int(TheXMLConfig.training('intervalLogEveryBatches', 0) or 0):
        return None
    started = time.perf_counter()
    result = dict(schema=1, event='forgetting',
        pass_number=int(model.symbolSpace.ltm_store.forget_passes),
        report=report,
        provisioned_truth_luminosity=provisioned_luminosity(model))
    result['collection_seconds'] = time.perf_counter() - started
    append_report(result)
    return result


def observe_interval(model, *, training, split):
    """One outer-loop call, after compaction and before periodic checkpointing."""
    from util import TheXMLConfig
    interval = int(TheXMLConfig.training('intervalLogEveryBatches', 0) or 0)
    step = int(getattr(model, '_train_batches_seen', 0))
    if not training or split != 'train' or interval <= 0 or step % interval:
        return None
    started = time.perf_counter()
    report = dict(schema=1, event='interval', batch=step, **captured_metrics(model))
    # Checkpoint collection is read-only. A child owns construction globals,
    # probe admissions, temporary memories, RNG draws and any evaluation work.
    with tempfile.TemporaryDirectory(prefix='basicmodel-interval-') as temporary:
        folder = Path(temporary)
        checkpoint, result = folder/'probe.ckpt', folder/'probes.json'
        model.save_weights(str(checkpoint))
        command = [sys.executable, str(Path(__file__).resolve()), '--config',
                   str(Path(model._config_path).resolve()), '--checkpoint', str(checkpoint),
                   '--output', str(result)]
        environment = dict(os.environ, BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false')
        with (folder/'probe.log').open('w+') as log:
            try:
                process = subprocess.run(command, env=environment, cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                log.seek(0)
                report['probes'] = dict(status='failed', reason='timeout', detail=log.read()[-12000:])
            else:
                if process.returncode:
                    log.seek(0)
                    report['probes'] = dict(status='failed', exit_code=process.returncode, detail=log.read()[-12000:])
                else:
                    report['probes'] = json.loads(result.read_text())
    report['collection_seconds'] = time.perf_counter() - started
    append_report(report)
    return report


def run_probes(config, checkpoint):
    import torch
    import util
    from CategoricalDiscrimination import FIXED_PROBES, fixed_probe_discrimination
    from eval_nanochat_grammar import build_eval_model, frozen_online_learning, _reset_model
    torch.set_num_threads(1)
    util.init_device('cpu')
    util.TheCompileBackend = 'none'
    model, data = build_eval_model(config, autoload=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data.source_manifest = saved['training_state']['data_manifest']
    probes = json.loads((ROOT/'data/interval-probes.json').read_text())
    readings, generic, both = {}, [], {'word': [], 'sentence': []}

    def read(text):
        _reset_model(model)
        with data.runtime_batch([text]), torch.no_grad():
            understanding = model.understand(model.inputSpace.prepInput([text]))
            fields = [field for group in understanding.sentence_fields.values() for field in group if field is not None]
            words = getattr(model, '_attention_words', None)
            if words is not None:
                pairs = words.poles[words.accepted].detach()
                both['word'].extend(((pairs[:, 0] > 0) & (pairs[:, 1] > 0)).cpu().tolist())
            both['sentence'].extend(field.evidence[0] > 0 and field.evidence[1] > 0 for field in fields)
            return understanding.symbolic_state.detach().cpu().flatten(), fields, understanding.input_reconstruction

    try:
        if not model.load_weights(str(checkpoint), require_match=True):
            raise ValueError('Probe checkpoint could not be restored')
        validate_probe_copy(model, saved)
        model._tensor_peer_while_eager = True
        model.eval()
        with frozen_online_learning(model):
            for name, panel in FIXED_PROBES.items():
                readings[name] = torch.stack([read(text)[0] for text in panel['texts']])
            for probe in probes['generic_subjects']:
                _, fields, _ = read(probe['text'])
                observed = [{"relation": "relative", "idea": "absolute"}.get(field.meaning.sentence_kind, "unknown") for field in fields]
                generic.append(dict(probe, observed=observed, correct=observed == [probe['expected']]))
            _, _, sample = read(probes['reconstruction'])
            reconstruction = dict(source=probes['reconstruction'], available=sample is not None)
            if sample is not None:
                reconstruction.update(byte_cost=scalar(sample.byte_cost),
                    truncated=sample.truncated.detach().cpu().tolist(),
                    reconstructed_event=sample.event.detach().cpu().tolist())
        return dict(status='complete', **fixed_probe_discrimination(readings), generic_subjects=generic,
            both_rate_by_bracket={level: dict(count=len(values), both=sum(values),
                rate=None if not values else sum(values)/len(values)) for level, values in both.items()},
            reconstruction=reconstruction, probe_counts={name: len(value) for name, value in readings.items()})
    finally:
        model.End()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_probes(args.config, args.checkpoint)
    result['probe_seconds'] = time.perf_counter() - started
    Path(args.output).write_text(json.dumps(result, allow_nan=False, sort_keys=True)+'\n')
