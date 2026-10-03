"""Observation-only pytest plugin; no source patch, RNG calls or gate changes."""
import json
import os
from pathlib import Path
import pytest


OBSERVER = None

def pytest_sessionstart(session):
    global OBSERVER
    path = os.environ.get('OWNERSHIP_OBSERVER_OUTPUT')
    if path is None:
        return
    import importlib.util, sys
    root = Path(__file__).resolve().parents[3]
    source = root/'test/objective_conflicts_probe.py'
    saved = sys.argv
    sys.argv = [str(source),'--config','XOR_grammar','--output',path,'--validate-only']
    spec = importlib.util.spec_from_file_location('ownership_observer', source)
    OBSERVER = importlib.util.module_from_spec(spec)
    try:
        try:
            spec.loader.exec_module(OBSERVER)
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = saved
    install_readback_gradient_probe()


def install_readback_gradient_probe():
    """Observe wrong read-backs on the actual training graph; never step it."""
    import torch
    from Models import BasicModel
    from SentenceUnderstanding import readback_scores
    original = BasicModel._byte_word_cost
    def observed(model, idea, word, codes, surfaces, validity, targets, target_valid,
                 ready, *, priming=None):
        result = original(model, idea, word, codes, surfaces, validity, targets,
                          target_valid, ready, priming=priming)
        probe = OBSERVER.P
        if (not ready or not getattr(model, '_sentence_training', False)
                or not probe.in_batch or not result.requires_grad):
            return result
        w = int(word)
        with torch.no_grad():
            present = validity.any(-1)
            weights = torch.ones_like(present, dtype=idea.dtype) if priming is None else priming
            scores = readback_scores(idea, codes, weights).masked_fill(~present, -torch.inf)
            chosen = scores.argmax(-1)
            def spelling(values, mask):
                text = []
                for value, valid in zip(values.tolist(), mask.tolist()):
                    if not valid or value == 0:
                        break
                    text.append(value)
                return text
            wrong = []
            active = model.inputSpace._word_active_mask[:, w].clone()
            leaf_mask = getattr(model.inputSpace, '_ar_grammar_leaf_mask', None)
            if torch.is_tensor(leaf_mask):
                active &= leaf_mask[:, w]
            ids = getattr(model.inputSpace, '_packed_sentence_ids', None)
            sid = getattr(model, '_open_sentence_slot', None)
            if torch.is_tensor(ids) and sid is not None:
                active &= ids[:, w] == sid
            for b, col in enumerate(chosen.tolist()):
                wrong.append(bool(active[b] and present[b].any()) and
                    spelling(surfaces[b,col], validity[b,col]) !=
                    spelling(targets[b,w], target_valid[b,w]))
            wrong = torch.tensor(wrong, device=idea.device)
        if bool(wrong.any()):
            named = [('leaf', idea), ('codes', codes)]
            live = [(name, value) for name, value in named if value.requires_grad]
            gradients = torch.autograd.grad(result[wrong].sum(), [value for _,value in live],
                                           retain_graph=True, allow_unused=True)
            rows = {}
            for (name, _), gradient in zip(live, gradients):
                rows[name] = None if gradient is None else gradient.detach().flatten(1).norm(dim=1).tolist()
            probe.log('wrong_readback_gradient', batch=probe.training_batches, epoch=probe.epoch,
                      trial=model._sentence_trial, word=w, wrong=wrong.tolist(),
                      cost=result.detach().tolist(), norms=rows,
                      scope='raw free-byte loss, before log-256 normalization; actual trial graph')
        return result
    BasicModel._byte_word_cost = observed


def save_ownership(model):
    if OBSERVER is None:
        return
    p = OBSERVER.P
    p.groups(model)
    p.geometry(model, "end")
    OBSERVER.write('configured-training.json', dict(
        OBSERVER.util.TheXMLConfig.get('architecture.training')))
    OBSERVER.write('outcome.json',dict(training_batches=p.training_batches,
        evaluation_batches=p.evaluation_batches,last_training_trials=p.last_trial,
        last_batch=p.last_batch,grammar_reconstructions=p.eval_reconstructions))
    assert OBSERVER.SOURCE == OBSERVER.source_snapshot(OBSERVER.ROOT)
    OBSERVER.write('complete.json',dict(source_matched=True,
        reused='predeclared first shared gate training; zero additional training runs'))


def record(value):
    with Path(os.environ['ITEM7_XOR_MEASUREMENTS']).open('a') as handle:
        value['gate'] = int(os.environ['ITEM7_XOR_GATE'])
        handle.write(json.dumps(value) + '\n')


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item):
    # The shared module fixture trains during setup, before either gate call.
    module = item.module
    grammar = getattr(module, '_run_xor_grammar_in_process', None)
    if grammar is not None:
        def observed_grammar(*args, **kwargs):
            model = grammar(*args, **kwargs)
            data = model.inputSpace.data
            save_ownership(model)
            record(dict(nodeid=item.nodeid, kind='grammar',
                        accuracy=[float(x) for x in model.rCorrect],
                        predictions=[float(x.reshape(-1)[0]) for x in data.reconstructed_output],
                        targets=[float(x.reshape(-1)[0]) for x in data.test_output],
                        inputs=[model._bytes_to_text(x).rstrip(chr(0)) for x in data.test_input],
                        reconstructions=list(data.reconstructed_input),
                        gate_reconstructions=list(model._grammar_gate_reconstructions),
                        grammar_reconstruction_unavailable=model._grammar_gate_unavailable))
            return model
        module._run_xor_grammar_in_process = observed_grammar
    try:
        yield
    finally:
        if grammar is not None:
            module._run_xor_grammar_in_process = grammar


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    module = item.module
    trained = getattr(getattr(item, 'instance', None), 'trained_xor_grammar', None)
    if trained is not None:
        record(dict(nodeid=item.nodeid, kind='shared_gate_consumer', model_identity=id(trained)))
    runner = getattr(module, 'run_config', None)
    if runner is not None:
        def observed_runner(*args, **kwargs):
            result = runner(*args, **kwargs)
            record(dict(nodeid=item.nodeid, kind='reconstruction', config=args[0],
                        epochs=kwargs.get('epochs'), seed=kwargs.get('seed'),
                        **{name: float(getattr(result, name)) for name in
                           ('exact_match_rate', 'where_recovery', 'output_loss', 'recon_loss')}))
            return result
        module.run_config = observed_runner
    cli = getattr(module, '_run_cli', None)
    if cli is not None:
        def observed_cli(*args, **kwargs):
            result = cli(*args, **kwargs)
            rc, stdout, stderr = result
            record(dict(nodeid=item.nodeid, kind='cli', config=args[0],
                        returncode=rc, stdout=stdout, stderr=stderr,
                        mse=module._parse_output_mse(stdout),
                        reconstructed=module._parse_input_match_counts(stdout)))
            return result
        module._run_cli = observed_cli
    mse_forward = None
    measurement = dict(nodeid=item.nodeid, kind='mm', calls=0, best=float('inf'))
    if item.name in ('test_convergence', 'test_learns_xor_signal',
                      'test_mm_grammar_learns_xor_signal'):
        import torch
        mse_forward = torch.nn.MSELoss.forward
        def observed_mse(self, output, target):
            result = mse_forward(self, output, target)
            if output.numel() == 4:
                value = float(result.detach())
                measurement.update(calls=measurement['calls'] + 1,
                                   best=min(measurement['best'], value), last=value,
                                   predictions=output.detach().cpu().flatten().tolist(),
                                   targets=target.detach().cpu().flatten().tolist())
            return result
        torch.nn.MSELoss.forward = observed_mse
    try:
        yield
    finally:
        if runner is not None:
            module.run_config = runner
        if cli is not None:
            module._run_cli = cli
        if mse_forward is not None:
            torch.nn.MSELoss.forward = mse_forward
            record(measurement)
