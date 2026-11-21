"""An explicit model-clock owner for isolated memory mechanisms."""
import torch
from types import MethodType


def install(model, tick=0):
    from Models import BasicModel
    from RowTime import encoder
    model.when_time = torch.tensor(tick, dtype=torch.long)
    model.when_code = encoder().encode(model.when_time).float()
    model.when_encoding = encoder()
    for name in ('present', '_advance_when_time', 'set_present', '_encode_present', 'present_code'):
        setattr(model, name, MethodType(getattr(BasicModel, name), model))
    model._sentence_ticks = {}
    if not hasattr(model, '_open_sentence_slot'):
        model._open_sentence_slot = None
    model._episode_counts = dict(written=0, anchors=0, relation_only_closings=0, time_only_recalls=0, reinstatements=0)
    owner = getattr(model, 'conceptualSpace', None)
    if owner is not None and hasattr(owner, 'stm'):
        object.__setattr__(owner.stm, '_time_owner', lambda: model)
    return model


def for_stm(stm, tick=0, model=None):
    from types import SimpleNamespace
    model = install(SimpleNamespace(_what_memory=lambda: None,
        conceptualSpace=SimpleNamespace(stm=stm)) if model is None else model, tick)
    object.__setattr__(stm, '_time_owner', lambda: model)
    return model
