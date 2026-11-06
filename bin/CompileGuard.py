"""Admit declared initial graphs, then reject model guard churn before fallback.

Each compiled callable owns its warmup record. A first call in a declared
training/evaluation, autograd, or decoder phase may specialize its graph. That
is initial compilation, not permission to resize tensors during the run.
Subsequent guard failures in that phase raise with Dynamo's original reason.
"""
from functools import wraps
import inspect
import re

import torch
from torch._dynamo import config
from torch._dynamo.exc import FailOnRecompileLimitHit, RecompileError


def reconstruction_phase(args, kwargs):
    """The reconstruction call's explicit greedy, departure and replay modes."""
    route = ('replay' if kwargs.get('decoder_kept_actions') is not None else
             'departure' if kwargs.get('decoder_exploit') is not None else 'greedy')
    return (route, kwargs.get('return_decoded', False), kwargs.get('decoder_candidates', False),
            kwargs.get('understanding') is not None, kwargs.get('candidate_basis') is not None,
            kwargs.get('keep_ideas') is not None)


def output_walk_phase(args, kwargs):
    """Declared trial modes and optional semantic banks, never tensor widths."""
    route = ('replay' if kwargs.get('kept_actions') is not None else
             'departure' if kwargs.get('exploit_actions') is not None else 'greedy')
    sample = kwargs.get('sample_actions', args[4] if len(args) > 4 else False)
    basis = kwargs.get('basis', args[5] if len(args) > 5 else None)
    banks = (basis is not None, *(kwargs.get(key) is not None for key in
             ('basis_priming', 'terminal_valid', 'case_bank', 'whole_bank',
              'constituents', 'constituent_valid', 'constituent_families', 'initial_depth')))
    return (route, kwargs.get('return_trace', False), kwargs.get('return_candidates', False),
            kwargs.get('require_symbols', True), sample,
            kwargs.get('structural_context', False), banks)


def percept_arrival_phase(args, kwargs):
    """The ownerless arrival brick's explicit training or observation mode."""
    return kwargs.get('training', args[12] if len(args) > 12 else False)


def compose_choice_phase(args, kwargs):
    """The native word loop and sentence close have distinct legal menus."""
    return kwargs.get('closing', False)


def carried_state_phase(args, kwargs):
    """A native word brick starts fresh, then carries its differentiable state."""
    return any(value.requires_grad for value in args[4])


def _cache_limit_description(error):
    """Read the actual limit from Dynamo's chained Unsupported diagnostic."""
    cause = error
    messages = []
    while cause is not None:
        message = str(cause)
        messages.append(message)
        match = re.search(r'Limit type: (accumulated_recompile_limit|recompile_limit)\b', message)
        if match:
            name = match.group(1)
            return f'{name}={getattr(config, name)}'
        cause = cause.__cause__
    # Preserve an unexpected upstream diagnostic rather than attributing the
    # failure to the wrong capacity or allowing eager fallback.
    return 'unidentified by Dynamo: ' + ' caused by '.join(messages)


def _owner(function):
    if isinstance(function, torch.nn.Module):
        return function
    bound = getattr(function, '__self__', None)
    if isinstance(bound, torch.nn.Module):
        return bound
    # Model word bricks and fixed-capacity loops close over their owning self.
    if inspect.isfunction(function) and function.__closure__:
        captured = dict(zip(function.__code__.co_freevars, function.__closure__, strict=True))
        if 'self' in captured and isinstance(captured['self'].cell_contents, torch.nn.Module):
            return captured['self'].cell_contents
    return None


def guarded_compile(function, *, initial_phase=None, **options):
    """Compile a model entry with per-callable, explicitly bounded warmup phases.

    Independent model instances and separately defined word-capacity sources
    each admit their initial graphs. The standard phases are module train/eval,
    grad-enabled/disabled and inference mode. Decoder callers additionally pass
    one of the named phase functions above. No tensor shape, stride, value or
    arbitrary owner attribute participates in phase admission.

    The policy lives only around this callable's execution; global Dynamo
    settings are restored even when compilation or execution raises. It also
    covers graph-break continuations when fullgraph=False. Inside a compiled
    caller the outer entry already owns the policy, so this wrapper does not
    trace a configuration context into the graph.
    """
    if options.get('backend') == 'none':
        return function
    owner = _owner(function)
    compiled = torch.compile(function, **options)
    invoke = compiled.forward if isinstance(compiled, torch.nn.Module) else compiled
    admitted = set()
    name = getattr(function, '__qualname__', type(function).__qualname__)

    @wraps(invoke)
    def checked(*args, **kwargs):
        if torch.compiler.is_compiling():
            return invoke(*args, **kwargs)
        phase = (None if owner is None else owner.training,
                 torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                 None if initial_phase is None else initial_phase(args, kwargs))
        # Fail on an invalid phase declaration rather than inventing a cache key.
        hash(phase)
        try:
            with config.patch(error_on_recompile=phase in admitted,
                              suppress_errors=False, fail_on_recompile_limit_hit=True):
                result = invoke(*args, **kwargs)
        except FailOnRecompileLimitHit as error:
            raise RuntimeError(f'Model compilation cache limit {_cache_limit_description(error)} in {name}; '
                               f'declared phase {phase}. {error}') from error
        except RecompileError as error:
            raise RuntimeError(f'Unexpected model recompilation in {name}; declared phase {phase}. '
                               f'{error}') from error
        admitted.add(phase)
        return result

    if isinstance(compiled, torch.nn.Module):
        # Preserve OptimizedModule identity, delegation and state-dict layout.
        object.__setattr__(compiled, 'forward', checked)
        return compiled
    return checked
