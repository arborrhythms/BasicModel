from pathlib import Path
import sys
ROOT=Path('/Users/arogers/github/WikiOracle/basicmodel')
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
from torch import nn
from Models import BasicModel
from test_compose_records import Runtime

def check(staged):
    model=Runtime()
    model.symbolSpace=nn.Module()
    model.symbolSpace.languageLayer=nn.Module()
    model.symbolSpace.languageLayer.operation_layer=nn.Identity()
    model.when_time=torch.tensor(1.)
    model.teacher._staged_source_rows=staged
    model.teacher._staged_split='train'
    def stage(split,rows):
        model.teacher._staged_source_rows=rows
        model.teacher._staged_split=split
    model.teacher.stage_batch_sources=stage
    calls=[]
    def once(*, train=True, split='train', exploration_trial=False, source_rows=None):
        calls.append(model.teacher._staged_source_rows)
        model.teacher._staged_source_rows=None
        model.teacher._staged_split=None
        return object(),0
    model._run_batch_once=once
    model._compose_state_snapshot=lambda:BasicModel._compose_state_snapshot(model)
    model._restore_compose_state=BasicModel._restore_compose_state
    model._compose_completed_rows=lambda:None
    model._exploration_constraints=lambda:(None,None,None)
    BasicModel._run_batch_pair(model,source_rows=[0,1])
    assert calls==[staged,staged], f'exploit/explore Teacher addresses differ: {calls!r}'
    assert model.teacher._staged_source_rows is None

for staged in (None, [[11],[12]]):
    check(staged)
