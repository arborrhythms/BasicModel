"""Structural graph-lifetime and optional-owner certificates for the repair."""
from dataclasses import fields, is_dataclass
import inspect
from types import SimpleNamespace
import torch
from test_item20_round34_projection import record


def tensors(value, path, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if torch.is_tensor(value):
        if value.grad_fn is not None:
            yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, path + '.' + str(key), seen)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, path + '.' + str(index), seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), path + '.' + field.name, seen)


def carried(model, sub, sym):
    seen = set()
    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if attr not in ('_parameters', '_modules'):
                yield from tensors(value, name + '.' + attr, seen)
    yield from tensors(sub, 'threaded.sub', seen)
    yield from tensors(sym, 'threaded.sym', seen)


def nodes(values):
    found, pending = set(), [v.grad_fn for v in values if v.grad_fn is not None]
    while pending:
        node = pending.pop()
        if node in found:
            continue
        found.add(node)
        pending.extend(parent for parent, _ in node.next_functions if parent is not None)
    return found


def test_packed_sentence_boundary_cuts_prior_writes_and_retains_perception(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from reading_fixtures import force_absolute_reading
    from What import What
    import SentenceTraining
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        concept_rows=256, part_rows=128)
    model._tensor_peer_while_eager = True
    model.reconstruction_placement = 'eager'
    model._chart_compose_per_word = lambda: None
    force_absolute_reading(model)
    model.inputSpace.data.has_supervised_outputs = False
    optimizer = model.getOptimizer(lr=.0001)
    original_begin, original_step = SentenceTraining.begin_sentence, model._sentence_train_step
    entry_nodes, sentence_nodes, first_choices, perception, scans, steps = set(), set(), [], [], [], []

    def begin(owner):
        frame = inspect.currentframe().f_back
        sub, sym = frame.f_locals['sub'], frame.f_locals['sym']
        values = list(carried(owner, sub, sym))
        if not steps:
            entry_nodes.update(nodes([v for _, v in values]))
            for space, names in ((owner.inputSpace, ('_ar_part_reference_vectors',
                    '_ar_whole_reference_vectors', '_ar_readout_coefficients')),
                    (owner.symbolSpace, ('_word_reference_presences',))):
                perception.extend(getattr(space, name) for name in names if torch.is_tensor(getattr(space, name)) and getattr(space, name).requires_grad)
        else:
            leaking = [name for name, value in values if nodes([value]) & sentence_nodes]
            scans.append(dict(tensors=len(values), leaks=leaking))
            record('boundary-scan', scans)
            assert not leaking, leaking
        return original_begin(owner)

    def step(loss):
        costs = model._single_sentence_objectives
        trace = model._reconstruction_stack()
        if not steps:
            # An explicit answer-side term makes graph retention structural,
            # independent of the untrained walk's sampled surface/advantage.
            costs['output'] = costs.get('output', 0) + trace._choice_values.square().sum()
            first_choices.extend(value for value in (trace._choice_values,
                trace._choice_action_records, trace._choice_log_probabilities) if value.requires_grad)
            frame = inspect.currentframe().f_back
            values = list(carried(model, frame.f_locals['sub'], frame.f_locals['sym']))
            sentence_nodes.update(n for n in nodes([v for _, v in values]) - entry_nodes
                                  if type(n).__name__ != 'AccumulateGrad')
        else:
            cost = sum(costs.values())
            old = torch.autograd.grad(cost, first_choices, allow_unused=True, retain_graph=True)
            assert all(value is None for value in old)
            live = torch.autograd.grad(cost, perception, allow_unused=True, retain_graph=True)
            assert any(value is not None for value in live), 'batch perception must remain reachable'
        result = original_step(loss)
        steps.append(1)
        return result

    monkeypatch.setattr(SentenceTraining, 'begin_sentence', begin)
    monkeypatch.setattr(model, '_sentence_train_step', step)
    try:
        raw = model.inputSpace.prepPackedInput([['a b', 'c d']])
        model.runBatch(train=True, optimizer=optimizer, batchSize=1,
            batch_override=(raw, torch.empty(1, 0)), questions=(What.present(0),))
        assert len(steps) == 2 and len(scans) == 1
        record('sentence-boundary', dict(optimizer_steps=2, first_sentence_A=True,
            scans=scans, old_choice_gradients='all None', batch_perception_reachable=True,
            retained_traversals=['R','A'], seeded=False))
    finally:
        model.End()


def test_separate_chooser_option_routes_compose_credit_without_reader_updates():
    from Layers import Error
    from Models import BaseModel, BasicModel
    logits = torch.nn.Parameter(torch.tensor([[.2, -.1, .4]]))
    reader = torch.nn.Parameter(torch.tensor([.5]))
    unrelated = torch.nn.Parameter(torch.tensor([.7]))
    model = torch.nn.Module()
    model.separate_chooser = True
    model.errors = Error()
    model._sentence_cost_registry = model.errors
    model._compose_forced_slots = torch.tensor([[True, False]])
    model._compose_sampling_scale = torch.tensor([[6., 0.]])
    model.objective_parameter_groups = lambda optimizer: {'chooser': (logits, unrelated), 'output': (reader,), 'attention': ()}
    actions = torch.tensor([[1, 0]])
    path = [None, [None] * 28]
    path[1][15], path[1][24] = actions, logits.log_softmax(-1).gather(1, actions)
    loss = BasicModel._compose_score_function_loss(model, path, torch.tensor([[1., .5]]))
    optimizer = torch.optim.SGD([logits, reader, unrelated], lr=.1)
    before = logits.detach().clone()
    BaseModel._backward_training_loss(model, loss, optimizer=optimizer)
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert reader.grad is None and unrelated.grad is None
    optimizer.step()
    assert logits.softmax(-1)[0, 1] > before.softmax(-1)[0, 1]
    record('separate-owner', dict(separateChooser=True, owner='chooser', side='A',
        before=before.tolist(), after=logits.detach().tolist(), reader_gradient=None))
