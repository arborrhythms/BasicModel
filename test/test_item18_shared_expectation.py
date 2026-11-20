"""intuit deposits its own map's intuited field; only the concept owner has the map."""
import pytest
import torch


def test_intuit_deposits_its_intuited_basic_block_then_closes(monkeypatch):
    from test_thought_infer import setup
    from Understanding import ConceptualField
    from SentenceField import publish, knowing
    from IntuitionField import open_field, read
    from ConceptSpread import open_prediction
    from QueryWork import QueryWorkBudget
    import FieldProjection
    model, _, _, refs = setup()
    owner = model._concept_owner()
    object.__setattr__(model, 'conceptualSpaces', (owner,))
    ids = torch.tensor([ref[1] for ref in refs])
    field = ConceptualField(ids, torch.zeros(len(ids), 1, 1, 2),
        torch.tensor([[[0, 1]]]), torch.tensor([[[0, 1]]]), None)
    width = owner._order_caps()[0]
    # A controlled map tests lane/row transport, independently of learning.
    # The stack gives one signed number for each concept; both signs occur.
    expected = (torch.arange(width).float()[None] - width / 2) / width
    monkeypatch.setattr(owner.intuit_map, 'forward', lambda value: expected)
    from Layers import signed_lanes
    intuited = signed_lanes(expected)
    publish(model, open_field(model, field))
    open_prediction(model, torch.tensor([True]))
    before_close = []
    close = FieldProjection.close
    def capture(space, rows, evidence, *args, **kwargs):
        before_close.append((rows, evidence.detach().clone()))
        return close(space, rows, evidence, *args, **kwargs)
    monkeypatch.setattr(FieldProjection, 'close', capture)
    read(owner, row=0, work=QueryWorkBudget(64))
    assert len(before_close) == 1
    rows, deposited = before_close[0]
    for index, address in enumerate(rows):
        if address < width:
            torch.testing.assert_close(deposited[index, 0, 0], intuited[0, address])
    assert not knowing(model).evidence.requires_grad
    assert not hasattr(model, '_intuition_teacher')


def test_only_the_concept_owner_builds_the_intuit_map_and_optimizer(tmp_path, monkeypatch):
    from test_math_chain import build_model
    model = build_model()
    try:
        owner = model._concept_owner()
        assert len(model.conceptualSpaces) == 3
        for space in model.conceptualSpaces:
            assert (space.intuit_map is not None) == (space is owner)
            assert not hasattr(space, 'intuit_stack')
        optimizer = model.getOptimizer(lr=.003)
        held = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert {id(p) for p in owner.intuit_map.parameters()} <= held
        assert not any('intuit_stack' in name for name, _ in model.named_parameters())
    finally:
        model.End(); model.symbolSpace.soft_reset()
