"""A third unreduced leaf must not erase the older leaf in an observer."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from probe_observation import full_choice

HERE = Path(__file__).resolve().parent
BIN = {'conjunction': torch.minimum, 'disjunction': torch.maximum}
tree = ast.parse((HERE / 'deriv69.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate')
exec(compile(ast.Module(body=[node], type_ignores=[]), str(HERE/'deriv69.py'), 'exec'))
calls = []
# Live STM is newest first; final tree is max(L0, min(L1, L2)).
for values, kind, op in (([3.],0,0),([5.,3.],0,0),([2.,5.,3.],1,0),([2.,3.],1,1)):
    state = (torch.tensor(values + [0.]*(4-len(values))).reshape(1,4,1), torch.tensor([len(values)]))
    choice = SimpleNamespace(kind=torch.tensor([kind]), local_op=torch.tensor([op]),
        position=torch.tensor([0]), valid=torch.tensor([True]))
    record = full_choice(state, choice, (['conjunction','disjunction'], ['not']))
    if '--old-window' in sys.argv:
        n = min(2, len(values))
        record.update(x=state[0][:,:n].flip(1).double(), depth=[n], pos=[0])
    calls.append(record)
stacks, leaves, deviations = simulate(calls)
assert deviations == [0.], deviations
assert len(leaves[0]) == 3, leaves
assert stacks[0][0][0] == 'max(L0,min(L1,L2))', stacks
assert float(stacks[0][0][1]) == 3.
print('Full STM observer preserves all three leaves and the final derivation.')
