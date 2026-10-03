"""Small mechanism checks; no model training campaign or fixture changes."""
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'test')]
import Models
import SentenceCompose
from probe_variants import install

variant = sys.argv[1]
install(variant, Models, SentenceCompose)

class Head:
    concept_ids = ()
    inputShape = (3, 2)
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(6))
    def __call__(self, sub):
        pred = sub.materialize().flatten(1) @ self.weight
        return SimpleNamespace(materialize=lambda: pred[:, None])

root = torch.tensor([[1., 2.], [2., 3.]], requires_grad=True)
slots = torch.cat((root[:, None], root.new_zeros(2, 2, 2)), 1)
head = Head()
model = SimpleNamespace(serial=True, outputSpace=head,
    normalizer=SimpleNamespace(denormalize=lambda value, **kwargs: value),
    inputSpace=SimpleNamespace(data=SimpleNamespace(has_supervised_outputs=True)),
    _probe_supplied_answers=torch.zeros(2, 1), _sentence_training=True,
    _align_output_pred=lambda pred, target: pred)
model._forward_head = MethodType(Models.BasicModel._forward_head, model)
lang = [None] * 15
lang[9], lang[13], lang[14] = root[:, None], slots.flatten(1)[:, None], torch.ones(2, 1, dtype=torch.long)
error = Models.BasicModel._probe_trial_answer_error(model, (None, lang, None), 0)
torch.testing.assert_close(error, torch.tensor([9., 25.]))
assert error.requires_grad == (variant == 'c')
if variant == 'c':
    error.sum().backward()
    assert root.grad is not None and root.grad.abs().sum() > 0
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
assert root.grad is None if variant != 'c' else True

parameter = torch.nn.Parameter(torch.tensor(2.))
optimizer = torch.optim.SGD([parameter], lr=.1)
parameters_seen, gradients = [], []
costs = [torch.tensor(row) for row in ([5., 5.], [0., 4.], [4., 3.], [3., 0.], [2., 2.])]
def compose(cache, prior):
    index = len(parameters_seen)
    parameters_seen.append(float(parameter.detach()))
    return parameter.square().expand(2) + costs[index], torch.full((2,), index)
def score(path, alternative):
    return path
def step(loss):
    optimizer.zero_grad()
    loss.backward()
    gradients.append(float(parameter.grad))
    optimizer.step()
with SentenceCompose.saved_sentence_values((parameter,)):
    selected, measured, winner = SentenceCompose.sentence_pair(
        None, compose, score, step, active=torch.ones(2, dtype=torch.bool))
n = 5 if variant == 'b' else 2
assert parameters_seen == [2.] * n
assert gradients == [4.] * n
assert selected.tolist() == ([1, 3] if variant == 'b' else [1, 1])
print(f'Variant {variant}: detached/uncut answer gradient, equal trial parameters, {n} separate updates, row-local winners verified.')
