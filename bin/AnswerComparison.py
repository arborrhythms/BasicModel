"""Independent answer parameters for loss-side comparison of sentence trials."""
from types import SimpleNamespace

import torch
from torch import nn


class _ReaderCall(nn.Module):
    """Expose the existing reader method to the public functional-call API."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model._sentence_reader_error(*args, **kwargs)


class AnswerComparison(nn.Module):
    """Same reading map, independent output-owned weights; never a public head.

    Copies are taken before either reader's first update. Allocation consumes
    no RNG and preserves tied parameter aliases. The temporary functional
    call restores all presented parameter bindings before returning.
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterDict()

    def sync(self, model):
        available = tuple(model.parameters())
        owned = model.objective_parameter_groups(
            SimpleNamespace(param_groups=[{'params': available}]))['output']
        comparison = {id(p) for p in self.parameters()}
        source = {id(p) for p in owned} - comparison
        fresh = model.__dict__.setdefault('_fresh_synthesis_params', [])
        for name, parameter in model.named_parameters():
            if id(parameter) not in source:
                continue
            key = name.replace('.', '/')
            if key not in self.weights:
                self.weights[key] = nn.Parameter(parameter.detach().clone())
                fresh.append(self.weights[key])

    def forward(self, model, *args, **kwargs):
        self.sync(model)
        copies = {id(model.get_parameter(name.replace('/', '.'))): value
                  for name, value in self.weights.items()}
        # Visit each physical module once. functional_call's automatic tied
        # expansion repeats setters for module aliases and can restore the
        # replacement rather than the original binding. Explicitly cover all
        # parameter attributes, including ties across distinct modules.
        parameters = {'model.' + (path + '.' if path else '') + name: copies[id(value)]
                      for path, module in model.named_modules()
                      for name, value in module.named_parameters(recurse=False, remove_duplicate=False)
                      if id(value) in copies}
        return torch.func.functional_call(_ReaderCall(model), parameters, args, kwargs,
                                          tie_weights=False)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        # The corresponding presented reader can have lazily allocated widths.
        # Restore the saved comparison bank without drawing an initialization.
        start = prefix + 'weights.'
        for name, value in state.items():
            if name.startswith(start):
                key = name[len(start):]
                if key not in self.weights:
                    self.weights[key] = nn.Parameter(torch.empty_like(value))
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
