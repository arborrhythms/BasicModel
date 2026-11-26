"""The fixes made after the review of the merge of items 1.5 and 1.6."""
from copy import deepcopy
from types import SimpleNamespace

import torch


def _table():
    from BasicPercepts import BasicPercepts
    table = BasicPercepts(None, 64, 64, 1024)
    table.bind(None)
    return table


def test_a_things_row_holds_no_primitive_property():
    from BasicPercepts import BasicPercepts
    # Row 1 is a thing: its whole is a seat of the table of values, below zero.
    basic = SimpleNamespace(index={i: i for i in range(3)}, things={1: ()},
        occurrences=[((), (0,)), ((), (-70,)), ((), (1,))], is_value=lambda row: False,
        whole=SimpleNamespace(primitive_properties=SimpleNamespace(on_counts=lambda counts: counts)))
    primitive = torch.ones(1, 3, 4, 2)
    rows = torch.tensor([[[0], [1], [2]]])
    result = BasicPercepts.property_presence(basic, primitive, rows, uses_wholes=True)
    assert result.shape[:2] == (1, 3)
    assert basic.__dict__['_property_members'][3].tolist() == [False, True, False, False]


def test_a_checkpoint_from_before_the_alphabet_loads():
    table, fresh = _table(), _table()
    state = {name: value for name, value in table.state_dict().items()
             if name not in ('levels', 'level_meets', 'level_seats')}
    fresh.load_state_dict(state)
    assert torch.equal(fresh.levels, table.levels)


def test_a_value_code_agrees_in_its_two_halves():
    table = _table()
    felt = torch.tensor(.5)
    code = torch.cat((table.value_code(felt, 64), table.value_whole(felt)))
    assert int(table.bin_of(code)) == int(table.value_bin(felt, 64))
    # The parts of the least value under the whole of the greatest: no value.
    crossed = torch.cat((table.value_code(torch.tensor(0.), 64), table.value_whole(torch.tensor(1.))))
    assert int(table.bin_of(crossed)) == -1
    # A range of values is the OR of the levels inside it, by their levels.
    low, high = torch.tensor(.25), torch.tensor(.75)
    whole = table.range_whole(low, high)
    for value, held in ((.25, True), (.5, True), (.75, True), (.1, False), (.9, False)):
        own = table.value_whole(torch.tensor(value))
        assert bool(torch.maximum(1 - own, whole).amin() == 1) is held, value


def test_a_copied_registry_keeps_a_built_grid_and_a_new_one_builds_none():
    from WhereRegistry import WhereRegistry
    registry = WhereRegistry([('text', 8), ('parts', 4)])
    assert 'grid' not in registry.__dict__ and 'grid' not in deepcopy(registry).__dict__
    grid = registry.grid
    copied = deepcopy(registry)
    assert 'grid' in copied.__dict__ and copied.grid is not grid and torch.equal(copied.grid, grid)
    assert registry._box_table(grid.device) is registry._box_table(grid.device)
