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
import os

import torch
import torch.optim as _optim


__all__ = [
    "Optimizer", "Adam", "SparseAdam", "RowLocalAdam", "MultiOptimizer",
    "finite_gradient_guard_enabled", "preflight_finite_gradients",
    "reconstruction_priority_gradient", "backward_reconstruction_priority",
    "configure_l1_proximal",
]


_FINITE_GRAD_GUARD_ENV = "MODEL_FINITE_GRAD_GUARD"
_FINITE_GRAD_GUARD_MODES = frozenset(("auto", "on", "off"))


def _optimizer_parameters(optimizer):
    """Yield each optimizer-owned parameter once, across all child groups."""
    seen = set()
    for group in getattr(optimizer, "param_groups", ()):
        for param in group.get("params", ()):
            identity = id(param)
            if identity in seen:
                continue
            seen.add(identity)
            yield param


def reconstruction_priority_gradient(reconstruction, output, *, max_ratio=0.5):
    """Project conflicting output credit away from reconstruction, then cap it.

    Applied independently to each protected parameter tensor: if r.o < 0,
    q = o - (r.o / r.r) r; otherwise q = o. Return q scaled so its norm is
    at most max_ratio * ||r||, with 0 <= max_ratio < 1. A missing/zero r
    permits no output contribution on that parameter. Output-only parameters
    should not be passed here. This is a first-order *gradient* contract;
    momentum, adaptive preconditioning and finite step sizes do not imply a
    monotonic reconstruction-loss guarantee.

    Sparse codebook gradients are aligned only over their touched COO entries,
    never over dictionary capacity. Mixed layouts can occur when another path
    already produces a dense gradient; only that case uses a dense result.
    """
    if not math.isfinite(max_ratio) or not 0.0 <= max_ratio < 1.0:
        raise ValueError("output gradient max_ratio must satisfy 0 <= ratio < 1")
    if output is None or reconstruction is None:
        return None
    r, o = reconstruction.detach(), output.detach()
    if r.shape != o.shape or r.device != o.device:
        raise ValueError("reconstruction/output gradients must match shape and device")
    if r.layout not in (torch.strided, torch.sparse_coo) or o.layout not in (
            torch.strided, torch.sparse_coo):
        raise ValueError("gradient projection supports dense and sparse COO only")
    sparse = r.is_sparse and o.is_sparse
    if sparse:
        r, o = r.coalesce(), o.coalesce()
        if r.sparse_dim() != o.sparse_dim():
            raise ValueError("sparse gradients must have matching sparse dimensions")
        indices, inverse = torch.unique(
            torch.cat((r.indices(), o.indices()), dim=1), dim=1,
            sorted=True, return_inverse=True)
        r_values = r.values().new_zeros(
            (indices.shape[1],) + tuple(r.values().shape[1:]))
        o_values = torch.zeros_like(r_values)
        r_values.index_add_(0, inverse[:r._nnz()], r.values())
        o_values.index_add_(0, inverse[r._nnz():], o.values())
    else:
        r_values = r.to_dense() if r.is_sparse else r
        o_values = o.to_dense() if o.is_sparse else o

    if r_values.numel() == 0:
        return o
    dtype = torch.float64 if r_values.dtype == torch.float64 else torch.float32
    rv, ov = r_values.to(dtype), o_values.to(dtype)
    # Normalize before taking dot products: squaring raw fp16/large fp32
    # gradients can overflow even though the projected answer is finite.
    r_max, o_max = rv.abs().amax(), ov.abs().amax()
    r_scaled = rv / torch.where(r_max > 0, r_max, 1.0)
    o_scaled = ov / torch.where(o_max > 0, o_max, 1.0)
    r_norm = torch.linalg.vector_norm(r_scaled)
    r_unit = r_scaled / torch.where(r_norm > 0, r_norm, 1.0)
    dot = (r_unit * o_scaled).sum()
    q = o_scaled - dot.clamp(max=0.0) * r_unit
    q_norm = torch.linalg.vector_norm(q)
    scale = torch.minimum(
        o_max,
        (max_ratio * r_max) * (
            r_norm / torch.where(q_norm > 0, q_norm, 1.0)))
    result = (q * scale).to(o_values.dtype)
    if sparse:
        return torch.sparse_coo_tensor(
            indices, result, size=o.shape, dtype=o.dtype, device=o.device,
            is_coalesced=True)
    return result


def backward_reconstruction_priority(total_loss, reconstruction_loss,
                                     output_loss, protected_parameters, *,
                                     max_ratio=0.5):
    """Populate .grad once per objective at one unchanged parameter version.

    The caller zeroes gradients first and performs ONE optimizer step later.
    Loss arguments include their actual weights and any truth modulation;
    total_loss includes output_loss exactly once. Scale all three equally
    when using AMP. Reconstruction is inspected with autograd.grad, output
    is differentiated, and the remaining objective is differentiated before
    adding the protected output contribution. Output-only heads keep their
    ordinary gradients, and auxiliary losses are unchanged.

    Do not subtract an enormous output gradient from an already-accumulated
    total .grad: cancellation can erase the smaller reconstruction gradient.
    Clear the protected output .grad before differentiating total - output.
    The partition is at the loss graph, before the upstream derivatives.
    """
    if not math.isfinite(max_ratio) or not 0.0 <= max_ratio < 1.0:
        raise ValueError("output gradient max_ratio must satisfy 0 <= ratio < 1")
    protected = list(dict.fromkeys(
        p for p in protected_parameters if p.requires_grad))
    if not protected or not output_loss.requires_grad:
        total_loss.backward()
        return
    if any(p.grad is not None for p in protected):
        raise RuntimeError("reconstruction-priority backward requires zeroed gradients")
    reference = (
        torch.autograd.grad(
            reconstruction_loss, protected, retain_graph=True, allow_unused=True)
        if reconstruction_loss.requires_grad else (None,) * len(protected))
    output_loss.backward(retain_graph=True)
    projected = [
        reconstruction_priority_gradient(r, p.grad, max_ratio=max_ratio)
        for p, r in zip(protected, reference)]
    for p in protected:
        p.grad = None
    (total_loss - output_loss).backward()
    for p, q in zip(protected, projected):
        if q is not None:
            p.grad = q if p.grad is None else p.grad + q


def _finite_gradient_guard_mode():
    mode = os.environ.get(_FINITE_GRAD_GUARD_ENV, "auto").strip().lower()
    if not mode:
        mode = "auto"
    if mode not in _FINITE_GRAD_GUARD_MODES:
        choices = "|".join(sorted(_FINITE_GRAD_GUARD_MODES))
        raise ValueError(
            f"{_FINITE_GRAD_GUARD_ENV} must be {choices}; got {mode!r}")
    return mode


def finite_gradient_guard_enabled(optimizer):
    """Return the finite-gradient policy for an optimizer's devices.

    ``auto`` protects CPU and MPS training.  It deliberately disables the
    check for the *complete* optimizer when any parameter is on CUDA: even a
    scalar result read would add a D2H synchronization to the captured brick.
    Set ``MODEL_FINITE_GRAD_GUARD=on`` to explicitly accept that CUDA cost.
    """
    mode = _finite_gradient_guard_mode()
    if mode != "auto":
        return mode == "on"
    return not any(
        getattr(getattr(param, "device", None), "type", None) == "cuda"
        for param in _optimizer_parameters(optimizer)
    )


def _gradient_fingerprint(optimizer):
    """Cheap host-metadata identity used for one runBatch->step handoff."""
    fingerprint = []
    for param in _optimizer_parameters(optimizer):
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        fingerprint.append((id(param), id(grad), int(grad._version)))
    return tuple(fingerprint)


def _parameter_name_map(named_parameters):
    if named_parameters is None:
        return {}
    names = {}
    for name, param in named_parameters:
        names.setdefault(id(param), str(name))
    return names


def preflight_finite_gradients(optimizer, named_parameters=None, *,
                               cache_for_step=False):
    """Validate every optimizer-owned gradient before any parameter update.

    Dense gradients are checked directly. Sparse COO gradients are coalesced
    first (so duplicate finite entries whose sum overflows are also caught),
    written back to ``param.grad``, and represented only by their stored
    values. Consequently a million-row sparse codebook costs O(touched rows),
    not O(codebook capacity), to inspect.

    Tensors are grouped by device and dtype and scanned with AMP's fused
    foreach non-finite primitive using an identity unscale. This incurs one
    scalar host read per populated group. A slow per-parameter name/count pass
    runs only after the fused scan reports a failure.

    ``cache_for_step`` records a metadata fingerprint for the optimizer
    wrappers. ``BaseModel.runBatch`` uses it to pass the successful preflight
    to ``Optimizer.step`` without performing the fused scan twice; any
    intervening in-place gradient mutation invalidates the fingerprint.
    """
    if not finite_gradient_guard_enabled(optimizer):
        if hasattr(optimizer, "_finite_grad_preflight_fingerprint"):
            optimizer._finite_grad_preflight_fingerprint = None
        return False

    names = _parameter_name_map(named_parameters)
    if names:
        optimizer._finite_grad_parameter_names = names
    else:
        names = getattr(optimizer, "_finite_grad_parameter_names", {})

    # (parameter, logical gradient values, tensor actually scanned, label)
    #
    # The AMP foreach primitive does not accept complex tensors.  Present
    # complex gradients through view_as_real instead of casting: both real
    # and imaginary components are then checked without allocation or loss of
    # information.  resolve_conj() makes the view valid for conjugate grads
    # while remaining a no-op for ordinary tensors.
    records = []
    groups = {}
    for ordinal, param in enumerate(_optimizer_parameters(optimizer)):
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if grad.layout == torch.sparse_coo:
            # Coalescing is part of the safety check: duplicate coordinates
            # can each be finite yet overflow while being summed.
            grad = grad.coalesce()
            param.grad = grad
            checked = grad.values()
        elif grad.layout == torch.strided:
            checked = grad
        else:
            raise RuntimeError(
                "finite-gradient preflight supports dense and sparse COO "
                f"gradients; got layout={grad.layout} for optimizer "
                f"parameter {ordinal}")
        if checked.numel() == 0:
            continue
        scanned = (torch.view_as_real(checked.resolve_conj())
                   if checked.is_complex() else checked)
        label = names.get(id(param), f"optimizer parameter {ordinal}")
        record = (param, checked, scanned, label)
        records.append(record)
        key = (scanned.device.type, scanned.device.index, scanned.dtype)
        groups.setdefault(key, []).append(record)

    failed_keys = []
    for key, group_records in groups.items():
        device = group_records[0][2].device
        found_inf = torch.zeros((), dtype=torch.float32, device=device)
        inv_scale = torch.ones((), dtype=torch.float32, device=device)
        torch._amp_foreach_non_finite_check_and_unscale_(
            [record[2] for record in group_records], found_inf, inv_scale)
        # This is the sole normal-path host synchronization for the group.
        if bool(found_inf.detach().cpu().item()):
            failed_keys.append(key)

    if failed_keys:
        failed = set(failed_keys)
        for _param, checked, scanned, label in records:
            key = (scanned.device.type, scanned.device.index, scanned.dtype)
            if key not in failed:
                continue
            nonfinite = ~torch.isfinite(checked)
            count = int(nonfinite.sum().detach().cpu().item())
            if count:
                raise FloatingPointError(
                    f"Non-finite gradient for {label!r} before optimizer.step: "
                    f"{count}/{checked.numel()} stored entries are nan/inf "
                    f"(layout={getattr(_param.grad, 'layout', None)}, "
                    f"device={checked.device}, dtype={checked.dtype}).")
        # The fused primitive is authoritative; retain fail-loud behavior even
        # if a backend's diagnostic reduction cannot reproduce its flag.
        raise FloatingPointError(
            "Non-finite gradient detected before optimizer.step; the "
            "per-parameter diagnostic pass could not isolate it")

    if cache_for_step:
        optimizer._finite_grad_preflight_fingerprint = (
            _gradient_fingerprint(optimizer))
    return True


def _consume_cached_finite_preflight(optimizer):
    cached = getattr(optimizer, "_finite_grad_preflight_fingerprint", None)
    if cached is None:
        return False
    optimizer._finite_grad_preflight_fingerprint = None
    return cached == _gradient_fingerprint(optimizer)


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
        # Per-batch observation policies, not durable optimizer state. The
        # caller stages these after zero_grad; skipped AMP steps cannot leak
        # a previous batch's reference mask into the next update.
        self._l1_proximal = {}

    def set_l1_proximal(self, param, rows, mask, strength):
        """Stage a row-local L1 prox for one update, after smooth gradients.

        ``strength`` already includes the caller's concept-count normalization.
        ``mask`` identifies penalized coefficients, normally excluding bias
        and unadmitted reference slots. No L1 subgradient may also be added
        to .grad: that would count the same objective twice.
        """
        strength = float(strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("proximal L1 strength must be finite and nonnegative")
        group = next((g for g in self.param_groups
                      if any(p is param for p in g["params"])), None)
        if group is None:
            raise ValueError("proximal L1 parameter is not owned by this optimizer")
        if not strength:
            self._l1_proximal.pop(param, None)
            return
        if group["maximize"] or group["eps"] <= 0:
            raise ValueError("proximal L1 requires minimizing Adam with positive epsilon")
        if (rows.dtype != torch.long or rows.ndim != 1
                or rows.device != param.device or mask.device != param.device
                or mask.dtype != torch.bool
                or tuple(mask.shape) != (rows.numel(), *param.shape[1:])):
            raise ValueError("proximal L1 requires long rows and a matching boolean coefficient mask")
        if (bool(((rows < 0) | (rows >= param.shape[0])).any())
                or bool((rows[1:] <= rows[:-1]).any())):
            raise ValueError("proximal L1 rows must be sorted, unique, and in bounds")
        if rows.numel():
            self._l1_proximal[param] = (rows.detach().clone(), mask.detach().clone(), strength)
        else:
            self._l1_proximal.pop(param, None)

    def zero_grad(self, set_to_none=True):
        self._l1_proximal.clear()
        return super().zero_grad(set_to_none=set_to_none)

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
        self._l1_proximal.clear()
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

        proximal, self._l1_proximal = self._l1_proximal, {}
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
                policy = proximal.get(param)
                if rows.numel() == 0 and policy is None:
                    continue
                values = grad.values()
                if tuple(values.shape[1:]) != tuple(param.shape[1:]):
                    raise RuntimeError(
                        "RowLocalAdam sparse value shape does not match the "
                        f"parameter tail: values={tuple(values.shape)}, "
                        f"parameter={tuple(param.shape)}")
                if group["maximize"]:
                    values = -values
                l1_mask = None
                if policy is not None:
                    observed_rows, observed_mask, l1_strength = policy
                    # Include observed definitions even at a zero smooth
                    # gradient (the L1 objective still exists). Unknown row-0
                    # placeholders are never admitted by the caller. A wholly
                    # disconnected parameter (grad=None above) stays untouched.
                    merged_rows = torch.unique(torch.cat((rows, observed_rows)), sorted=True)
                    merged_values = values.new_zeros((merged_rows.numel(), *param.shape[1:]))
                    merged_values.index_add_(0, torch.searchsorted(merged_rows, rows), values)
                    l1_mask = torch.zeros_like(merged_values, dtype=torch.bool)
                    l1_mask.index_copy_(
                        0, torch.searchsorted(merged_rows, observed_rows), observed_mask)
                    rows, values = merged_rows, merged_values
                # Masked unknown concept ids are gathered through a safe row-0
                # placeholder and multiplied by zero in forward. Autograd can
                # still emit a structurally present COO row whose complete
                # value is exactly zero. Treat it as absent unless explicitly
                # observed by the L1 policy: letting a placeholder reach
                # Adam would decay old row-0 moments and move that real concept
                # despite the unknown identity carrying no gradient.
                live = values.reshape(values.shape[0], -1).ne(0).any(dim=1)
                if l1_mask is not None:
                    live = live | l1_mask.reshape(l1_mask.shape[0], -1).any(dim=1)
                    l1_mask = l1_mask[live]
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
                if l1_mask is not None:
                    # Diagonal-metric proximal Adam: with D=sqrt(v_hat)+eps,
                    # minimize .5/eta * ||z-y||_D^2 + strength*|z|_1, where
                    # y=theta-eta*m_hat/D. Its coordinate threshold is
                    # eta*strength/D, NOT eta*strength and NOT step_size
                    # (m_hat's first-moment bias correction is already in y).
                    # This is the L1 prox specialized to a positive diagonal
                    # metric; see arxiv.org/abs/1910.10094 and Parikh/Boyd §6.5.2.
                    threshold = (group["lr"] * l1_strength) / denom
                    shrunk = updated.sign() * (updated.abs() - threshold).clamp_min(0)
                    updated = torch.where(l1_mask, shrunk, updated)
                param.index_copy_(0, rows, updated.to(dtype=param.dtype))
        return loss


def configure_l1_proximal(optimizer, param, rows, mask, strength):
    """Find the sole row-local owner through ordinary/MultiOptimizer wrappers."""
    leaves = getattr(optimizer, "optimizers", (optimizer,))
    owners = [leaf for leaf in leaves if any(
        p is param for group in leaf.param_groups for p in group["params"])]
    if len(owners) != 1:
        raise ValueError("proximal L1 requires exactly one optimizer owner")
    inner = getattr(owners[0], "inner", owners[0])
    if not isinstance(inner, _RowLocalAdam):
        raise TypeError("concept readout L1 requires its RowLocalAdam optimizer")
    inner.set_l1_proximal(param, rows, mask, strength)


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

    def _step_without_finite_preflight(self, closure=None):
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

    def step(self, closure=None):
        """Reject non-finite gradients before the wrapped optimizer mutates.

        Optimizer closures create the gradients that the step consumes, so a
        closure must run before the finite-gradient preflight.  Execute it
        exactly once under grad mode, then call the inner optimizer without a
        closure; standard torch optimizer semantics return that closure loss.
        """
        if closure is not None:
            # A cached runBatch preflight necessarily predates this closure.
            self._finite_grad_preflight_fingerprint = None
            with torch.enable_grad():
                loss = closure()
            preflight_finite_gradients(self)
            self._step_without_finite_preflight()
            return loss
        if not _consume_cached_finite_preflight(self):
            preflight_finite_gradients(self)
        return self._step_without_finite_preflight()

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
    working, a merged ``state`` view keyed by parameter (``optimizer.state[p]``
    as on a torch optimizer), and ``add_param_group`` for heads built lazily
    after construction. ``state_dict()`` keeps the per-child ``optimizers``
    layout the checkpoint remapper consumes and adds torch-style ``state`` /
    ``param_groups`` views (state keyed by the flattened parameter index).
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self._refresh_param_groups()

    def _refresh_param_groups(self):
        flat = []
        for o in self.optimizers:
            flat.extend(o.param_groups)
        self.param_groups = flat

    @property
    def state(self):
        """Merged parameter -> state view over every child optimizer."""
        return _MergedOptimizerState(self.optimizers)

    def _dense_child(self):
        """The child that owns ordinary dense parameters (never the sparse
        or row-local family): the first child without those markers."""
        for o in self.optimizers:
            if (getattr(o, "row_local_state", False)
                    or type(o).__name__ == "SparseAdam"):
                continue
            return o
        return self.optimizers[0]

    def add_param_group(self, param_group):
        """Hand a late-built head to the dense child (torch semantics)."""
        self._dense_child().add_param_group(param_group)
        self._refresh_param_groups()

    def _step_without_finite_preflight(self):
        results = []
        for o in self.optimizers:
            unchecked = getattr(o, "_step_without_finite_preflight", None)
            if unchecked is not None:
                results.append(unchecked())
            else:
                results.append(o.step())
        return results

    def step(self, closure=None):
        """Preflight the union of child grads before the first child step.

        A shared closure is evaluated exactly once.  All child gradients are
        then checked together and every child is stepped without receiving the
        closure, preventing both repeated evaluation and partial commits.
        """
        if closure is not None:
            # A cached runBatch preflight necessarily predates this closure.
            self._finite_grad_preflight_fingerprint = None
            with torch.enable_grad():
                loss = closure()
            preflight_finite_gradients(self)
            self._step_without_finite_preflight()
            return loss
        if not _consume_cached_finite_preflight(self):
            preflight_finite_gradients(self)
        return self._step_without_finite_preflight()

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        children = [o.state_dict() for o in self.optimizers]
        # torch-style views: state keyed by the FLATTENED parameter index
        # (child offsets applied), param_groups with re-indexed ``params``.
        merged_state = {}
        merged_groups = []
        offset = 0
        for child in children:
            for key, value in (child.get("state") or {}).items():
                merged_state[offset + int(key)] = value
            for group in child.get("param_groups") or ():
                shifted = dict(group)
                shifted["params"] = [offset + int(i) for i in group.get("params", ())]
                merged_groups.append(shifted)
            offset += sum(len(g.get("params", ())) for g in child.get("param_groups") or ())
        return {"optimizers": children, "state": merged_state,
                "param_groups": merged_groups}

    def load_state_dict(self, state):
        children = state.get("optimizers")
        if children is not None:
            for o, s in zip(self.optimizers, children):
                o.load_state_dict(s)
            self._refresh_param_groups()
            return
        # torch-style dict: split back by the children's parameter counts.
        merged_state = state.get("state") or {}
        merged_groups = list(state.get("param_groups") or [])
        offset = 0
        cursor = 0
        for o in self.optimizers:
            n_groups = len(o.param_groups)
            groups = merged_groups[cursor:cursor + n_groups]
            cursor += n_groups
            count = sum(len(g.get("params", ())) for g in groups)
            local_groups = []
            for g in groups:
                lg = dict(g)
                lg["params"] = [int(i) - offset for i in g.get("params", ())]
                local_groups.append(lg)
            local_state = {int(k) - offset: v for k, v in merged_state.items()
                           if offset <= int(k) < offset + count}
            o.load_state_dict({"state": local_state, "param_groups": local_groups})
            offset += count
        self._refresh_param_groups()


class _MergedOptimizerState:
    """Read-through ``optimizer.state`` over MultiOptimizer children."""

    def __init__(self, optimizers):
        self._optimizers = optimizers

    def _owner(self, param):
        for o in self._optimizers:
            if param in o.state:
                return o
        return None

    def __contains__(self, param):
        return self._owner(param) is not None

    def __getitem__(self, param):
        owner = self._owner(param)
        if owner is None:
            raise KeyError(param)
        return owner.state[param]

    def get(self, param, default=None):
        owner = self._owner(param)
        return default if owner is None else owner.state[param]

    def items(self):
        for o in self._optimizers:
            yield from o.state.items()

    def keys(self):
        for o in self._optimizers:
            yield from o.state.keys()

    def values(self):
        for o in self._optimizers:
            yield from o.state.values()

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(len(o.state) for o in self._optimizers)
