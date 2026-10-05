"""Observation-only plugin for test_grammar_word_learning.py's unchanged test."""
import json
import os
from pathlib import Path
import torch

def emit(kind, **values):
    def convert(value):
        if torch.is_tensor(value): return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)
    with Path(os.environ['REVIEW15_CHOOSER_DIAG']).open('a') as handle:
        handle.write(json.dumps(dict(kind=kind, **values), default=convert)+'\n')

def pytest_sessionstart(session):
    from Models import BasicModel
    original = BasicModel._sentence_path_cost
    def scored(model, state, sid, active):
        value = original(model, state, sid, active)
        emit('cost', trial=model._sentence_trial, active=active,
             cost=model._sentence_cost_registry.total(objective='reconstruction'),
             actions=state[1][18], rules=state[1][4], logp=state[1][27],
             forced=model._compose_forced_slots,
             binary=[r.method_name for r in model.languageSpace._compose_binary_rules],
             unary=[r.method_name for r in model.languageSpace._compose_unary_rules])
        return value
    BasicModel._sentence_path_cost = scored
    preference = BasicModel._compose_preference_loss
    def compared(model, path, wins):
        value = preference(model, path, wins)
        emit('preference', penalty=value, **model._last_compose_preference)
        return value
    BasicModel._compose_preference_loss = compared
