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

    Both derivations are scored before either optimizer update. Their
    backwards and shared perception must read the original parameters and buffers. A shared weight may be
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


def compile_word_brick(function):
    """Capture a fixed-shape word step; the sentence/optimizer boundary is eager."""
    import util
    backend = util.TheCompileBackend
    if backend == 'none':
        return function
    backend = util._COMPILE_BACKENDS[0] if backend == 'auto' else backend
    return torch.compile(function, backend=backend, fullgraph=True)


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


def sentence_pair(cache, compose, score, step, *, active, training=True):
    """Probe b: greedy plus four independent deviations, all scored before training."""
    exploit = compose(cache, None)
    cost, state = score(exploit, False)
    if cost.shape != active.shape:
        raise ValueError('sentence comparison requires one cost per row')
    costs, states = [cost], [state]
    if training:
        for _ in range(4):
            cost, state = score(compose(cache, exploit), True)
            if cost.shape != active.shape:
                raise ValueError('sentence comparison requires one cost per row')
            costs.append(cost)
            states.append(state)
    saved = torch.stack([cost.detach().clone() for cost in costs], -1)
    # argmin retains the earliest trial on a tie, including greedy at index 0.
    winner = torch.where(active, saved.argmin(-1), torch.zeros_like(active, dtype=torch.long))
    if training:
        for cost in costs:
            step((cost * active.to(cost)).sum() / active.sum().clamp_min(1))
    selected = states[0]
    for index, state in enumerate(states[1:], 1):
        selected = select_rows(selected, state, active & (winner == index))
    return selected, saved, winner
