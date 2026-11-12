"""Eager sentence endings around fixed-shape numerical compose bricks."""
import torch
from contextlib import contextmanager
from weakref import WeakValueDictionary
from torch._subclasses.fake_tensor import is_fake


class _SavedValue:
    def __init__(self, value, storage, copy):
        # Holding storage prevents its address being reused while a saved
        # value is live, without retaining the original tensor's graph.
        self.storage = storage
        self.value = value.clone() if copy else value.detach()
        self.version = None if copy else value._version

    def unpack(self):
        if self.version is not None and self.value._version != self.version:
            raise RuntimeError('a sentence intermediate was mutated before backward')
        return self.value


@contextmanager
def saved_sentence_values(parameters=(), buffers=()):
    """Keep one immutable saved value per tensor view and parameter version.

    Packed sentences share a perception forward across their optimizer steps.
    Their backwards must read its original parameters and buffers. A shared weight may be
    saved hundreds of times by the word/operator rounds; copying every save
    would multiply dictionary memory by the number of uses.
    """
    protected = {value.untyped_storage()._cdata for values in (parameters, buffers)
                 for value in values if value.layout == torch.strided}
    saved = WeakValueDictionary()

    def pack(value):
        # HOP backward capture constructs symbolic placeholders here. They
        # have no optimizer-owned runtime storage to freeze, and symbolic
        # sizes are deliberately unhashable. Let the traced graph own them.
        if is_fake(value):
            return _SavedValue(value, None, False)
        if value.layout != torch.strided:
            return _SavedValue(value, None, True)
        storage = value.untyped_storage()
        base = value if value._base is None else value._base
        if isinstance(base, torch.nn.Parameter):
            protected.add(storage._cdata)
        key = (storage._cdata, value._version, value.storage_offset(),
               tuple(value.shape), value.stride(), value.dtype)
        result = saved.get(key)
        if result is None:
            # Functional scratch and MLP intermediates are immutable. Retain
            # them directly and check versions when read; only optimizer-owned
            # storage needs a frozen copy across the two steps.
            result = _SavedValue(value, storage, storage._cdata in protected)
            saved[key] = result
        return result

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.unpack()):
        yield


def compile_word_brick(function, *, initial_phase=None):
    """Capture a fixed-shape word step; the sentence/optimizer boundary is eager."""
    import util
    backend = util.TheCompileBackend
    if backend == 'none':
        return function
    backend = util._COMPILE_BACKENDS[0] if backend == 'auto' else backend
    from CompileGuard import guarded_compile
    return guarded_compile(function, initial_phase=initial_phase,
                           backend=backend, fullgraph=True, dynamic=False)


def fork_perception(cache):
    """Give a compose trial fresh leaves, with an exact pullback to perception.

    Its backward can free the compose graph immediately. Only the single
    perception graph survives the first optimizer update for the second path
    and any batch-end answer objective.
    """
    leaves = {}
    def visit(value):
        if torch.is_tensor(value):
            if not value.requires_grad:
                return value
            key = id(value)
            if key not in leaves:
                leaves[key] = (value, value.detach().requires_grad_())
            return leaves[key][1]
        return type(value)(visit(item) for item in value)
    trial = visit(cache)
    def pullback():
        pairs = [(original, leaf.grad) for original, leaf in leaves.values()
                 if leaf.grad is not None]
        if pairs:
            originals, gradients = zip(*pairs)
            torch.autograd.backward(originals, gradients, retain_graph=True)
    def gradients(cost, parameters):
        """Read direct and perception gradients without touching .grad buffers."""
        from GradientDiagnostics import _sum_gradients
        pairs = tuple(leaves.values())
        measured = torch.autograd.grad(cost,
            (*parameters, *(leaf for _, leaf in pairs)), retain_graph=True, allow_unused=True)
        direct = measured[:len(parameters)]
        used = [(original, grad) for (original, _), grad in
                zip(pairs, measured[len(parameters):]) if grad is not None]
        if not used:
            return direct
        originals, cotangents = zip(*used)
        indirect = torch.autograd.grad(originals, parameters, grad_outputs=cotangents,
                                       retain_graph=True, allow_unused=True)
        return tuple(_sum_gradients(a, b) for a, b in zip(direct, indirect))
    pullback.gradients = gradients
    return trial, pullback


def select_rows(exploit, explore, wins):
    """Swap scratch tensor rows; no blended path and no model snapshot."""
    if torch.is_tensor(exploit):
        if exploit.ndim == 0:
            raise ValueError('sentence state must retain its row dimension')
        return torch.where(wins.reshape((-1,) + (1,) * (exploit.ndim - 1)),
                           explore, exploit)
    if isinstance(exploit, tuple):
        return tuple(select_rows(a, b, wins) for a, b in zip(exploit, explore))
    if isinstance(exploit, dict):
        return {k: select_rows(v, explore[k], wins) for k, v in exploit.items()}
    raise TypeError(f'unsupported sentence scratch state: {type(exploit).__name__}')


@torch.no_grad()
def kept_reading_plan(wins, paths, attention, fields, width):
    """Fixed actions and field visits for the winning stream of each reading."""
    actions = torch.where(wins[:, None], paths[-1], paths[0])
    reading = attention[0]
    if reading is not None:
        parts = select_rows(reading, attention[-1], wins)
        reading = type(reading)(type(reading.table)(*parts[0]), *parts[1:])
    reads = []
    if fields[0] is not None:
        left, right = fields[0].reads, fields[-1].reads
        exhausted = torch.full_like(actions[:, 0], width)
        for turn in range(max(len(left), len(right))):
            a = left[turn]['action'] if turn < len(left) else exhausted
            b = right[turn]['action'] if turn < len(right) else exhausted
            reads.append(torch.where(wins, b, a))
    return dict(actions=actions, attention=reading, field=reads)
