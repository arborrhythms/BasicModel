"""Disjoint writers and fixed expectation evidence."""
import torch


def test_expectation_does_not_write_its_context_or_routing():
    from Layers import IntraSentenceLayer
    layer = IntraSentenceLayer(4, 3, routing_dim=2)
    source = torch.randn(2, 3, 4, requires_grad=True)
    routing = torch.randn(2, 2, requires_grad=True)
    layer(source, routing=routing).square().sum().backward()
    assert source.grad is None
    assert routing.grad is None
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in layer.parameters())


def test_objectives_write_only_their_owners_even_with_shared_graphs():
    from ObjectiveOwnership import backward_owned
    representation, predictor, reader = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    costs = {'reconstruction': representation,
             'expectation': 100 * representation + 2 * predictor,
             'output': 100 * representation + 3 * reader}
    owners = {'reconstruction': (representation,), 'expectation': (predictor,), 'output': (reader,)}
    backward_owned(costs, owners)
    assert representation.grad.item() == 1
    assert predictor.grad.item() == 2
    assert reader.grad.item() == 3


def test_reconstruction_pullback_cannot_credit_an_answer_parameter():
    from ObjectiveOwnership import backward_owned
    from SentenceCompose import fork_perception
    percept = torch.nn.Parameter(torch.tensor(2.))
    reader = torch.nn.Parameter(torch.tensor(3.))
    (leaf,), pullback = fork_perception((percept * 2,))
    backward_owned({'reconstruction': leaf, 'output': -leaf + reader},
                   {'reconstruction': (percept,), 'expectation': (), 'output': (reader,)},
                   pullback=pullback)
    assert percept.grad.item() == 2
    assert reader.grad.item() == 1


def test_answer_vocabulary_alias_keeps_perceptual_owner():
    from types import SimpleNamespace
    from Models import BasicModel
    from ObjectiveOwnership import backward_owned
    vocabulary = torch.nn.Linear(2, 2, bias=False)
    output = torch.nn.Linear(2, 1, bias=False)
    output.add_module('_vocabulary', vocabulary)
    model = SimpleNamespace(outputSpace=output, synthesis_parameters=lambda: ())
    optimizer = torch.optim.SGD(output.parameters(), lr=.1)
    groups = BasicModel.objective_parameter_groups(model, optimizer)
    assert any(p is vocabulary.weight for p in groups['reconstruction'])
    assert not any(p is vocabulary.weight for p in groups['output'])
    backward_owned({'reconstruction': vocabulary.weight.sum(),
                    'output': output.weight.sum() + 7 * vocabulary.weight.sum()}, groups)
    torch.testing.assert_close(vocabulary.weight.grad, torch.ones_like(vocabulary.weight))
    torch.testing.assert_close(output.weight.grad, torch.ones_like(output.weight))


def test_named_concept_evidence_has_the_answer_as_its_only_writer():
    from pathlib import Path
    from recon_bench import _build_model
    import Models
    model, _, _, _ = _build_model(str(Path(Models.__file__).resolve().parents[1]/'data/XOR_exact.xml'))
    try:
        from ConceptLessons import teach_concept_lessons
        teach_concept_lessons(model)
        optimizer=model.getOptimizer(lr=.01)
        groups=model.objective_parameter_groups(optimizer)
        coefficients=model.conceptualSpaces[0]._concept_allocator.layer(0).values
        assert any(p is coefficients for p in groups['output'])
        assert not any(p is coefficients for p in groups['reconstruction'])
    finally:
        model.End()


def test_trial_consumers_share_one_record_and_answer_cannot_write_it(monkeypatch):
    from pathlib import Path
    import Models
    import util
    from test_mm_xor import _fresh_model
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    root = Path(Models.__file__).resolve().parents[1]
    model, _, data = _fresh_model(str(root/'data/XOR_grammar.xml'))
    reconstructed, answered = [], []
    reconstruct = model._reconstruct_trial
    answer = Models.BasicModel._sentence_answer_error
    def read(record):
        reconstructed.append(record)
        return reconstruct(record)
    def score(self, state, sid, active, observation, **kwargs):
        record = observation['record']
        answered.append(record)
        cost = answer(self, state, sid, active, observation, **kwargs)
        assert torch.autograd.grad(cost.sum(), record.root, retain_graph=True, allow_unused=True)[0] is None
        return cost
    monkeypatch.setattr(model, '_reconstruct_trial', read)
    monkeypatch.setattr(Models.BasicModel, '_sentence_answer_error', score)
    optimizer = model.getOptimizer(lr=.01)
    try:
        raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
        batch = model.inputSpace.prepInput(raw), model.outputSpace.prepOutput(target)
        model.runBatch(train=True, optimizer=optimizer, batchSize=4, split='train', batch_override=batch)
        assert len(reconstructed) == len(answered) == 2
        assert all(a is b for a, b in zip(reconstructed, answered))
        assert reconstructed[0].primed is reconstructed[1].primed
        assert reconstructed[0].primed.rows.shape[0] == 4
        assert model._word_symbol_concept_ids() is None
        report = model.ownership_gradient_diagnostics(optimizer)
        assert report['conflicts'] == 0
        assert {r['owner'] for r in report['parameters'] if r['writers']} == {
            'reconstruction', 'expectation', 'output'}
        assert all(not row['writers'] or row['writers'] == [row['owner']] for row in report['parameters'])
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
