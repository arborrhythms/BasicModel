"""Read saved round-2d data only: no model construction, RNG, or training.

The least-squares calculations diagnose representation geometry. They do
not replace any measured prediction or reproduce the full reader optimizer.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parent
ROOT = RECEIPT.parents[2]
INPUTS = {}


def read(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def unit(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def affine_fit(values, targets):
    centered = values - values.mean(0)
    # Minimize coefficient norm with a free intercept; do not penalize bias.
    weights = np.linalg.lstsq(centered, targets - targets.mean(), rcond=1e-10)[0]
    bias = targets.mean() - values.mean(0) @ weights
    singular = np.linalg.svd(centered, compute_uv=False)
    return dict(slope_norm=float(np.linalg.norm(weights)), bias=float(bias),
                mse=float(np.mean((values @ weights + bias - targets) ** 2)),
                affine_rank=int(np.linalg.matrix_rank(
                    np.c_[values, np.ones(len(values))], tol=1e-10)),
                centered_singular_values=singular.tolist())


def main():
    summary = read(RECEIPT / 'measurements/summary.json')
    aggregate = read(RECEIPT / 'aggregate-audit.json')
    policies = {r['run']: r for r in aggregate['xor']['runs']}
    runs = []
    for run in summary['xor']:
        audit = read(RECEIPT / f"measurements/xor-{run['run']:02}/run-audit.json")
        roots = np.array(audit['end']['roots']['values'])
        codebank = audit['end']['codes'][1]
        codes = dict(zip(codebank['rows'], np.array(codebank['values']), strict=True))
        rows = [r['word_rows'] for r in run['final_greedy_compose']]
        left = unit(np.stack([codes[pair[0]] for pair in rows]))
        right = unit(np.stack([codes[pair[1]] for pair in rows]))
        # Layers.Ops kernels at unit presence, as recorded in this gate.
        conjunction = unit(left * right)
        disjunction = unit(left + right - left * right)
        error = float(np.abs(conjunction - roots).max())
        assert error < 2e-7
        assert audit['end']['root_inputs'] == run['inputs']
        target = np.array(run['targets'])
        single = affine_fit(roots, target)
        joint = affine_fit(np.r_[roots, disjunction], np.r_[target, target])
        assert single['affine_rank'] == 4 and single['mse'] < 1e-20
        assert joint['mse'] < 1e-20
        # Use saved normalized answer costs directly, without assuming their
        # conversion to raw MSE. Compare the same cost at each epoch window.
        trials = [t for t in audit['sentence_trials'] if t['training']]
        costs = {str(epoch): float(np.mean([
            np.array(t['components'])[:, 0, 2].mean() for t in trials
            if epoch - 19 <= t['epoch'] <= epoch])) for epoch in (100, 200, 300, 400)}
        names = ('inputSpace.outputSpace.layers.0.W', 'answer_record_reader.weight')
        norms = {str(epoch): float(np.linalg.norm([
            audit['reader_weights'][epoch - 1]['parameters'][name] for name in names]))
            for epoch in (350, 400)}
        runs.append(dict(run=run['run'], measured_mse=run['mse'],
            measured_class_pass=run['class_pass'],
            last_greedy_disjunction_epoch=policies[run['run']]['last_greedy_disjunction_epoch'],
            reconstructed_conjunction_max_error=error,
            conjunction_only_affine_fit=single, both_roots_affine_fit=joint,
            joint_to_single_slope_norm=joint['slope_norm'] / single['slope_norm'],
            greedy_answer_cost_last_twenty_epochs=costs,
            active_reader_matrix_norms=norms,
            reader_norm_growth_last_fifty_epochs=norms['400'] / norms['350'] - 1))
    result = dict(
        method='Linear algebra and aggregation of the original saved runs only.',
        gate_trainings_added=0, source_changes=0, runs=runs,
        settled_reader_objective=dict(
            condition='Greedy conjunction; narrowing leaves the same root; compose offers disjunction.',
            round2c=dict(conjunction=5/6, disjunction=1/6),
            round2d=dict(conjunction=3/4, disjunction=1/4),
            explanation='Reader weights are not corrected for the changed departure proposal; the chooser surrogate is.'),
        measured_mse_vs_smallest_nonzero_conjunction_singular_value_correlation=float(
            np.corrcoef([r['measured_mse'] for r in runs],
                        [r['conjunction_only_affine_fit']['centered_singular_values'][2]
                         for r in runs])[0, 1]),
        limitations=[
            'Fits use fixed root vectors and a free affine intercept, not the full reader feature bank or Adam.',
            'Additional reader features can change conditioning; joint root fit norms are diagnostic, not full-model bounds.',
            'Exact fits establish representability, not convergence within the unchanged 400-epoch budget.',
            'Unpaired campaigns do not identify the causal share of sampling, native-code geometry, and optimization.',
        ], input_sha256=INPUTS,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in INPUTS.items())
    (HERE / 'reader_geometry.json').write_text(json.dumps(result, indent=2) + '\n')
    failures = [r for r in runs if not r['measured_class_pass']]
    print(json.dumps(dict(
        all_conjunction_roots_affinely_fit=True,
        joint_to_single_slope_norm_range=[min(r['joint_to_single_slope_norm'] for r in runs),
                                         max(r['joint_to_single_slope_norm'] for r in runs)],
        failed_run_reader_norm_growth_last_fifty_epochs=[
            min(r['reader_norm_growth_last_fifty_epochs'] for r in failures),
            max(r['reader_norm_growth_last_fifty_epochs'] for r in failures)],
        failed_runs_with_conjunction_throughout=[r['run'] for r in failures
                                               if r['last_greedy_disjunction_epoch'] == 0],
        inputs_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
