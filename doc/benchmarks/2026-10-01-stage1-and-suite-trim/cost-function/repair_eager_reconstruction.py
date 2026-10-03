"""Honor the selected no-compile backend for reconstruction traversal only."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'suite-trim'));from port_ledger import definitions
p=ROOT/'bin/Models.py';s=p.read_text();old=definitions(s)['BasicModel::_reconstruct_sentences']
new=old.replace('= torch.while_loop(', '= _reconstruction_while_loop(')
assert new.count('= _reconstruction_while_loop(')==3
helper='''def _reconstruction_while_loop(condition, body, values):
    """Run the same traversal body under the selected execution backend.

    torch.while_loop captures even when the enclosing model is eager, and
    its uncached backward reconstructs a graph on every trial. With compile
    disabled, ordinary autograd can differentiate these short functional
    loops directly. Explicit compiled execution keeps the existing HOP.
    """
    import util
    if torch.compiler.is_compiling() or util.TheCompileBackend != 'none':
        return torch.while_loop(condition, body, values)
    while bool(condition(*values)):
        values = body(*values)
    return values


'''
(HERE/'eager-reconstruction-repair.json').write_text(json.dumps(dict(
 failing_probes=['eager-loop-before/worker-000.log','before-eager-reconstruction/XOR_grammar-step5a/run.log','before-eager-reconstruction/interrupted.json'],
 changes=[dict(name='_reconstruction_while_loop',old=None,new=helper),dict(name='BasicModel._reconstruct_sentences',old=old,new=new)],
 cost_or_configuration_changed=False),indent=2)+'\n')
s=s.replace(old,new,1).replace('def _release_loop_checkpoints(roots=()):',helper+'def _release_loop_checkpoints(roots=()):',1);p.write_text(s)
p=ROOT/'test/test_mixing_reconstruction_bank.py';s=p.read_text();s+='''\n\ndef test_eager_reconstruction_loop_keeps_captured_values_and_gradients(monkeypatch):
    import util
    from Models import _reconstruction_while_loop
    results = []
    for backend in ('none', 'eager'):
        monkeypatch.setattr(util, 'TheCompileBackend', backend)
        x = torch.tensor([1.5, -.7], requires_grad=True)
        weight = torch.tensor([.8, 1.2], requires_grad=True)
        def condition(i, value):
            return i < 3
        def body(i, value):
            return i + 1, value * weight
        _, result = _reconstruction_while_loop(condition, body, (torch.tensor(0), x))
        gradients = torch.autograd.grad(result.sum(), (x, weight))
        torch.testing.assert_close(result, x * weight ** 3)
        torch.testing.assert_close(gradients[0], weight ** 3)
        torch.testing.assert_close(gradients[1], 3 * x * weight ** 2)
        results.append((result, gradients))
    torch.testing.assert_close(results[0], results[1])
    torch._dynamo.reset()
''';p.write_text(s)
