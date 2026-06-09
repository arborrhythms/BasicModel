"""Project optimizer wrappers -- the single seam between basicmodel
training code and ``torch.optim``. Every Adam / SparseAdam construction
in the project goes through ``bin.Optimizer`` rather than ``torch.optim``
directly, so backend-specific workarounds and policy live in exactly
one place.

Why the wrapper exists
======================
``bin/util.py:init_device`` calls
``torch.set_default_device(str(TheDevice.get()))`` at startup. On Apple
Silicon that pins the process-wide default device to ``mps``, which has
two known fallout modes for ``torch.optim``:

1. **Upstream pytorch/pytorch#149184** -- "instantiating optimizer after
   ``torch.set_default_device('mps')`` throws ``RuntimeError: Placeholder
   storage has not been allocated on MPS device!``" Reproduced against
   SGD and AdamW on PyTorch 2.7; Apple/PyTorch is aware. Stock
   ``Adam.__init__`` does not allocate scalars itself, so it dodges the
   direct repro, but is on the same root path.

2. **Silent ``step_t += 1`` drop in the radix byte path.** With default
   device ``mps``, non-capturable ``Adam`` lazy-inits its scalar
   ``state["step"]`` with ``torch.tensor(0.0)`` inside ``_init_group``;
   the kwarg-less constructor honors the default device so the scalar
   lands on ``mps:0``. In our radix forward path the MPS command queue
   gets into a state where the on-device 0-dim integer increment is
   silently dropped (the same corruption class makes the ``isfinite``
   reduction on the loss false-positive). The very next batch hits
   ``bias_correction1 = 1 - beta1**0 = 0`` and the line
   ``step_size = lr / bias_correction1`` in
   ``torch/optim/adam.py:_single_tensor_adam`` raises
   ``ZeroDivisionError``. Full investigation in
   ``doc/plans/2026-06-08-radix-mps-adam-step-zero.md``.

The wrapper temporarily restores ``cpu`` as the default device for the
duration of ``step()``. That matches stock CUDA Adam's layout: scalar
``state["step"]`` lives on CPU; dense moments (``exp_avg`` /
``exp_avg_sq``) still land on the parameter's device because they're
allocated via ``torch.zeros_like(param)``, not ``torch.zeros``. Because
``Adam.__init__`` does not allocate scalars (they're created lazily in
``step``, ``adam.py:168-176``), wrapping ``step`` alone is sufficient.
The guard is a thread-local default-device flip; CUDA / CPU runs see
no behavioural change beyond a redundant cpu->cpu set when the default
already is cpu.

What this module exports
========================
- ``Optimizer``: thin wrapper over any ``torch.optim.Optimizer``
  instance. Implements the full ``torch.optim.Optimizer`` surface that
  the project uses (``step``, ``zero_grad``, ``state_dict`` /
  ``load_state_dict``, ``param_groups``, ``state``, ``defaults``,
  ``add_param_group``) by explicit delegation, with the MPS-safe
  default-device guard applied around ``step``.

- ``Adam`` / ``SparseAdam``: convenience constructors that build the
  underlying ``torch.optim.Adam`` / ``torch.optim.SparseAdam`` and
  return a wrapped ``Optimizer``. Signatures match the upstream
  constructors so existing code can switch ``torch.optim.Adam(...)`` ->
  ``Optimizer.Adam(...)`` with no other change.

- ``MultiOptimizer``: combines several ``Optimizer`` instances behind
  one interface (the sparse + dense pair the codebook training uses).
  Replaces the inline ``_MultiOptimizer`` that previously lived in
  ``bin/Models.py``.

Known limits
------------
- A checkpoint saved by stock ``Adam`` under default-device ``mps``
  has ``state["step"]`` already on MPS;
  ``Optimizer.load_state_dict`` does NOT move it for non-capturable /
  non-fused Adam (``torch/optim/optimizer.py:799-803``). Loading such
  a checkpoint into the wrapper bypasses the guard for the first
  step after resume. Mitigation if you have such checkpoints: after
  ``load_state_dict``, force ``state[p]["step"]`` to CPU.
- ``fused=True`` and ``capturable=True`` follow different code paths
  inside ``_init_group`` (they allocate ``state["step"]`` with
  ``device=p.device``, ``adam.py:168-170``) and are NOT covered by
  this wrapper; the MPS workaround is moot there because the step
  already lands on the param device by design.
"""

import torch
import torch.optim as _optim


__all__ = ["Optimizer", "Adam", "SparseAdam", "MultiOptimizer"]


class Optimizer:
    """Thin wrapper over a ``torch.optim.Optimizer`` that applies the
    MPS-safe default-device guard around ``step``.

    Composition (not inheritance) so any optimizer type can be wrapped
    uniformly. ``isinstance(opt, torch.optim.Optimizer)`` returns False
    on wrappers; this is by design -- the project never uses that check,
    and the wrapper presents the methods the project actually uses.
    """

    def __init__(self, inner):
        if not isinstance(inner, _optim.Optimizer):
            raise TypeError(
                f"Optimizer wrapper expects a torch.optim.Optimizer, "
                f"got {type(inner).__name__}"
            )
        self._inner = inner

    # ------------------------------------------------------------------ step

    def step(self, closure=None):
        """Run ``inner.step`` with default device pinned to CPU.

        See module docstring for the MPS rationale. The flip is a
        thread-local op and is a no-op when the default device is
        already ``cpu`` (CPU runs, or non-MPS targets that were never
        flipped by ``init_device``).
        """
        prev = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            if closure is None:
                return self._inner.step()
            return self._inner.step(closure=closure)
        finally:
            torch.set_default_device(prev)

    # --------------------------------------------------------- forwarding API

    def zero_grad(self, set_to_none=True):
        return self._inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        return self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        return self._inner.add_param_group(param_group)

    @property
    def param_groups(self):
        return self._inner.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._inner.param_groups = value

    @property
    def state(self):
        return self._inner.state

    @property
    def defaults(self):
        return self._inner.defaults

    @property
    def inner(self):
        """Read-only access to the wrapped ``torch.optim.Optimizer``.

        Useful for code that needs the concrete class (AMP ``GradScaler``
        introspection, custom schedulers, debugging) -- in that case
        pass ``opt.inner`` to the consumer.
        """
        return self._inner


def Adam(params, lr, **kwargs):
    """Construct an ``Optimizer`` wrapping ``torch.optim.Adam``.

    Drop-in replacement for ``torch.optim.Adam(params, lr=lr, **kwargs)``;
    accepts the same ``params`` shapes (list, generator, list-of-param-
    groups) and the same kwargs (``betas``, ``eps``, ``weight_decay``,
    ``amsgrad``, ``capturable``, ``fused``, etc.).
    """
    return Optimizer(_optim.Adam(params, lr=lr, **kwargs))


def SparseAdam(params, lr, **kwargs):
    """Construct an ``Optimizer`` wrapping ``torch.optim.SparseAdam``.

    ``SparseAdam`` keeps ``state["step"]`` as a Python ``int`` (not a
    tensor), so it is NOT affected by the MPS scalar-step issue the
    module docstring describes. Wrapping it uniformly anyway means the
    project never touches the raw ``torch.optim`` namespace and the
    ``MultiOptimizer`` below can hold a homogeneous list.
    """
    return Optimizer(_optim.SparseAdam(params, lr=lr, **kwargs))


class MultiOptimizer:
    """Run several ``Optimizer`` instances behind one interface.

    Concrete use: the basicmodel codebook split puts sparse-grad
    embedding rows under ``SparseAdam`` and everything else under
    ``Adam`` (``BaseModel.getOptimizer`` -> ``MultiOptimizer([dense,
    sparse])``). The wrapper exposes a flattened ``param_groups``
    view so callers that read or set
    ``param_groups[i]['lr']`` (rebuild_optimizer, LR scheduling) keep
    working.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        flat = []
        for o in self.optimizers:
            flat.extend(o.param_groups)
        self.param_groups = flat

    def step(self, closure=None):
        results = []
        for o in self.optimizers:
            if closure is None:
                results.append(o.step())
            else:
                results.append(o.step(closure=closure))
        return results

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        for o, s in zip(self.optimizers, state.get("optimizers", [])):
            o.load_state_dict(s)
