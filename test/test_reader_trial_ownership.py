"""Discarded trial rows do not train, including Adam momentum updates."""
import torch
from Layers import Error
from ObjectiveOwnership import registry_costs, backward_owned


def test_reader_backpropagates_only_kept_rows_and_skips_an_empty_trial():
    reader=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    optimizer=torch.optim.Adam([reader],lr=.01)
    (reader.square().sum()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    registry=Error(row_mask=torch.ones(3,dtype=torch.bool))
    registry.squared('answer',reader,torch.ones(3),objective='output')
    costs=registry_costs(registry,reader_rows=torch.tensor([True,False,True]))
    backward_owned(costs, {'output':(reader,)})
    expected=2*(reader.detach()-1)/3
    expected[1]=0
    torch.testing.assert_close(reader.grad,expected)
    optimizer.zero_grad(set_to_none=True)
    before=reader.detach().clone()
    costs=registry_costs(registry,reader_rows=torch.zeros(3,dtype=torch.bool))
    assert 'output' not in costs
    backward_owned(costs, {'output':(reader,)})
    optimizer.step()
    torch.testing.assert_close(reader,before,rtol=0,atol=0)


def test_generation_lesson_merge_trains_reconstruction_rows():
    values=torch.nn.Parameter(torch.tensor([2.,3.]))
    registry=Error(row_mask=torch.ones(2,dtype=torch.bool))
    for row in range(2):
        lesson=Error()
        lesson.squared('generate.lesson',values[row],torch.tensor(1.),category='grammar')
        registry.merge(lesson,row=row)
    costs=registry_costs(registry,reader_rows=torch.tensor([False,True]))
    backward_owned(costs, {'generate_lesson':(values,)})
    torch.testing.assert_close(values.grad,torch.tensor([1.,2.]))  # both reconstruction trials train
