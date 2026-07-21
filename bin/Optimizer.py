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

import math

import torch
import torch.optim as _optim


__all__ = [
    "Optimizer", "Adam", "SparseAdam", "RowLocalAdam", "MultiOptimizer",
]


class _RowLocalAdam(_optim.Optimizer):
    """Adam over sparse row gradients with compact prefix moments.

    ``torch.optim.SparseAdam`` skips arithmetic on untouched rows, but still
    allocates full-size ``exp_avg`` and ``exp_avg_sq`` tensors.  That is not a
    viable distinction for the million-row ConceptualSpace dictionary: its two
    nominally sparse moments would consume another ~8 GiB at fp32.

    This optimizer accepts only sparse COO gradients whose sparse dimension is
    parameter axis 0.  Its two moments cover ``[0, R)`` where ``R`` grows to
    the next power of two above the largest row ever touched, capped by the
    physical parameter size.  Concept rows in the aligned serial model are
    allocated monotonically from zero, so the compact prefix follows actual
    occupancy while preserving stable row ids.  The state tensors use ordinary
    optimizer state_dict machinery and therefore checkpoint without a custom
    sidecar. ``moment_dtype=torch.bfloat16`` halves persistent moment storage
    while keeping Adam arithmetic in fp32 on each touched row; bfloat16 is
    used instead of fp16 because its exponent range does not underflow
    ordinary small squared gradients.
    """

    row_local_state = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, maximize=False,
                 moment_dtype=torch.float32):
        if not 0.0 <= float(lr):
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= float(eps):
            raise ValueError(f"invalid epsilon value: {eps}")
        if not 0.0 <= float(betas[0]) < 1.0:
            raise ValueError(f"invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= float(betas[1]) < 1.0:
            raise ValueError(f"invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"invalid weight_decay value: {weight_decay}")
        if moment_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "RowLocalAdam moment_dtype must be torch.float32 or "
                f"torch.bfloat16; got {moment_dtype}")
        defaults = dict(
            lr=float(lr), betas=tuple(float(v) for v in betas),
            eps=float(eps), weight_decay=float(weight_decay),
            maximize=bool(maximize), moment_dtype=moment_dtype,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _grow_prefix_state(param, state, required_rows, moment_dtype):
        physical_rows = int(param.shape[0])
        required = int(required_rows)
        if required < 1 or required > physical_rows:
            raise RuntimeError(
                "RowLocalAdam sparse row is outside the parameter: "
                f"required={required}, physical={physical_rows}")
        old_rows = int(state["exp_avg"].shape[0])
        if required <= old_rows:
            return old_rows
        target = min(physical_rows, 1 << (required - 1).bit_length())
        tail = tuple(param.shape[1:])
        for name in ("exp_avg", "exp_avg_sq"):
            old = state[name]
            grown = torch.zeros(
                (target, *tail), device=param.device, dtype=moment_dtype)
            if old_rows:
                grown[:old_rows].copy_(old.to(dtype=moment_dtype))
            state[name] = grown
        return target

    def load_state_dict(self, state_dict):
        """Restore compact moments and re-establish their storage dtype.

        PyTorch's generic optimizer loader casts floating state tensors to the
        parameter dtype. That would silently inflate bfloat16 row-local
        moments back to fp32 after resume, so recast the two known moments
        after the ordinary parameter-id remap.
        """
        configured_dtypes = [
            group.get("moment_dtype", torch.float32)
            for group in self.param_groups]
        result = super().load_state_dict(state_dict)
        for group_index, group in enumerate(self.param_groups):
            # Checkpoints from the initial fp32 RowLocalAdam implementation
            # predate this param-group field. In that one migration case, use
            # the newly-constructed optimizer's configured storage dtype.
            if group.get("moment_dtype") is None:
                group["moment_dtype"] = configured_dtypes[group_index]
            moment_dtype = group["moment_dtype"]
            if moment_dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(
                    "loaded RowLocalAdam moment_dtype must be torch.float32 "
                    f"or torch.bfloat16; got {moment_dtype}")
            for param in group["params"]:
                state = self.state.get(param)
                if not state:
                    continue
                for name in ("exp_avg", "exp_avg_sq"):
                    value = state.get(name)
                    if torch.is_tensor(value):
                        state[name] = value.to(
                            device=param.device, dtype=moment_dtype)
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            moment_dtype = group.get("moment_dtype", torch.float32)
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if param.ndim < 1:
                    raise RuntimeError(
                        "RowLocalAdam requires a parameter with a row axis")
                if not grad.is_sparse:
                    raise RuntimeError(
                        "RowLocalAdam requires sparse COO gradients; a dense "
                        "gradient would defeat its memory guarantee")
                grad = grad.coalesce()
                indices = grad.indices()
                if indices.ndim != 2 or int(indices.shape[0]) != 1:
                    raise RuntimeError(
                        "RowLocalAdam supports sparsity only on parameter "
                        f"axis 0; gradient indices shape={tuple(indices.shape)}")
                rows = indices[0].long()
                if rows.numel() == 0:
                    continue
                values = grad.values()
                if tuple(values.shape[1:]) != tuple(param.shape[1:]):
                    raise RuntimeError(
                        "RowLocalAdam sparse value shape does not match the "
                        f"parameter tail: values={tuple(values.shape)}, "
                        f"parameter={tuple(param.shape)}")
                if group["maximize"]:
                    values = -values
                # Masked unknown concept ids are gathered through a safe row-0
                # placeholder and multiplied by zero in forward. Autograd can
                # still emit a structurally present COO row whose complete
                # value is exactly zero. Treat it as absent: letting it reach
                # Adam would decay old row-0 moments and move that real concept
                # despite the unknown identity carrying no gradient.
                live = values.reshape(values.shape[0], -1).ne(0).any(dim=1)
                rows = rows[live]
                values = values[live]
                if rows.numel() == 0:
                    continue
                # Persistent moments may be bfloat16, but every touched-row
                # update (including eps and bias correction) is fp32.
                values = values.float()

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    tail = tuple(param.shape[1:])
                    state["exp_avg"] = torch.zeros(
                        (0, *tail), device=param.device,
                        dtype=moment_dtype)
                    state["exp_avg_sq"] = torch.zeros(
                        (0, *tail), device=param.device,
                        dtype=moment_dtype)
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(name)
                    if (not torch.is_tensor(moment)
                            or moment.ndim != param.ndim
                            or tuple(moment.shape[1:])
                            != tuple(param.shape[1:])
                            or int(moment.shape[0]) > int(param.shape[0])
                            or moment.dtype != moment_dtype):
                        raise RuntimeError(
                            f"RowLocalAdam state {name!r} has invalid shape "
                            f"{None if not torch.is_tensor(moment) else tuple(moment.shape)} "
                            f"or dtype {getattr(moment, 'dtype', None)} "
                            f"for parameter {tuple(param.shape)}")

                # This scalar synchronization is deliberately optimizer-side,
                # outside the compiled forward/backward graph.  No full
                # parameter or moment tensor crosses to the host.
                required = int(rows.max().item()) + 1
                self._grow_prefix_state(
                    param, state, required, moment_dtype)
                state["step"] = int(state.get("step", 0)) + 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                if group["weight_decay"] != 0.0:
                    values = values.add(
                        param.index_select(0, rows),
                        alpha=group["weight_decay"])
                avg_rows = exp_avg.index_select(0, rows).float()
                sq_rows = exp_avg_sq.index_select(0, rows).float()
                avg_rows.mul_(beta1).add_(values, alpha=1.0 - beta1)
                sq_rows.mul_(beta2).addcmul_(
                    values, values, value=1.0 - beta2)
                exp_avg.index_copy_(
                    0, rows, avg_rows.to(dtype=moment_dtype))
                exp_avg_sq.index_copy_(
                    0, rows, sq_rows.to(dtype=moment_dtype))

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = group["lr"] / bias_correction1
                denom = (sq_rows.sqrt()
                         / math.sqrt(bias_correction2)).add_(group["eps"])
                updated = param.index_select(0, rows).float()
                updated.addcdiv_(avg_rows, denom, value=-step_size)
                param.index_copy_(0, rows, updated.to(dtype=param.dtype))
        return loss


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

        The restore goes through the CANONICAL ``util.TheDevice`` string,
        NOT a ``torch.get_default_device()`` snapshot: torch normalizes
        ``'mps'`` to ``device('mps', index=0)`` on read, and the two are
        NOT ``==``. Restoring the normalized snapshot therefore changed
        the ambient-device IDENTITY that ``torch.compile`` guards on
        (``utils_device.CURRENT_DEVICE == device('mps')`` failed after the
        first optimizer step), forcing a full retrace (~200-380s on MPS).
        ``init_device`` keeps ``TheDevice`` and the default device in
        sync, so this restores exactly the spelling the process set.
        """
        from util import TheDevice  # local: Optimizer imports before util elsewhere
        torch.set_default_device("cpu")
        try:
            if closure is None:
                return self._inner.step()
            return self._inner.step(closure=closure)
        finally:
            torch.set_default_device(str(TheDevice.get()))

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
    def row_local_state(self):
        """Whether moment tensors intentionally cover only a row prefix."""
        return bool(getattr(self._inner, "row_local_state", False))

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


def RowLocalAdam(params, lr, **kwargs):
    """Construct compact-prefix Adam for sparse row gradients.

    Unlike :class:`torch.optim.SparseAdam`, optimizer moments are not sized to
    the complete parameter.  See :class:`_RowLocalAdam` for the checkpoint and
    geometric-growth contract.
    """
    return Optimizer(_RowLocalAdam(params, lr=lr, **kwargs))


class MultiOptimizer:
    """Run several ``Optimizer`` instances behind one interface.

    Concrete use: the basicmodel codebook split puts sparse-grad embedding
    rows under ``SparseAdam``, the shared ConceptualSpace dictionary under
    ``RowLocalAdam``, and everything else under ``Adam``. The wrapper exposes
    a flattened ``param_groups`` view so callers that read or set
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
