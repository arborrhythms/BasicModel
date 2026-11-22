"""A sparse LDU with both triangles, in tests (item 1.7, round 6d, seventh pass,
step 1); the model's layer is :class:`Layers.SparseLDULayer`.

W = L D U in :class:`Layers.InvertibleLinearLayer`'s row-vector form
(``x @ L @ D @ U``) and with its padding: L unit lower over the inputs, D a
full vector of the rank, U unit upper over the outputs. The strict triangles
are entries, each an index pair and a value. The slot axis is the first.

* The way up is scatters: ``x @ (I + N)`` adds ``x[i] * v`` into ``j`` for
  each entry ``(i, j, v)``.
* The way back is exact: ``(I + N)^-1 = I - N + N^2 - ...`` ends after as
  many terms as the longest chain of entries in N, so ``x <- y - x @ N``,
  started at ``x = y``, that many times, undoes ``I + N``.
"""
import torch

from Layers import unit_apply as _apply, unit_solve as _solve


def unit_apply(x, entries):
    """``x @ (I + N)``, ``N`` the entries ``(source, target, value)`` (:func:`Layers.unit_apply`)."""
    return _apply(x, *entries)


def unit_solve(y, entries, steps):
    """``x`` with ``x @ (I + N) = y`` (:func:`Layers.unit_solve`)."""
    return _solve(y, *entries, steps)


def longest_chain(source, target):
    """The longest chain of entries (edges) from a source to a target; raises
    on a cycle, where no number of steps would end the series."""
    succ = {}
    for s, t in zip(torch.as_tensor(source).tolist(), torch.as_tensor(target).tolist()):
        succ.setdefault(s, []).append(t)
    memo, state = {}, {}

    def down(node):
        if node in memo:
            return memo[node]
        if state.get(node) == 'open':
            raise ValueError('a cycle of entries')
        state[node] = 'open'
        result = 1 + max((down(t) for t in succ.get(node, ())), default=-1)
        state[node] = 'done'
        memo[node] = result
        return result

    return max((down(s) for s in list(succ)), default=0)


def ldu_up(x, d, lower, upper, n_output):
    """``x @ L @ D @ U``, ``x`` ``[nInput, ...]``, padded as InvertibleLinearLayer pads."""
    rank = len(d)
    shape = (-1,) + (1,) * (x.ndim - 1)
    z = unit_apply(x, lower)[:rank] * d.reshape(shape)
    if n_output > rank:
        z = torch.cat((z, z.new_zeros(n_output - rank, *z.shape[1:])))
    return unit_apply(z, upper)


def ldu_back(y, d, lower, upper, n_input, depth_lower, depth_upper):
    """The exact way back, the pseudoinverse that the padding gives."""
    rank = len(d)
    shape = (-1,) + (1,) * (y.ndim - 1)
    z = unit_solve(y, upper, depth_upper)[:rank] / d.reshape(shape)
    if n_input > rank:
        z = torch.cat((z, z.new_zeros(n_input - rank, *z.shape[1:])))
    return unit_solve(z, lower, depth_lower)


def dense_w(layer):
    """The dense W ``[nInput, nOutput]`` of a :class:`Layers.SparseLDULayer`
    from its factors, ``L D U`` over its slots, differentiable in its
    learned numbers: the definition the scatters of its way up must give."""
    S, n = layer.slots, layer.nnz
    value = layer.value[:n]
    kind, source, target = layer.entry_kind[:n], layer.entry_source[:n], layer.entry_target[:n]
    eye = torch.eye(S, dtype=value.dtype)

    def triangle(k):
        keep = kind == k
        return eye.index_put((source[keep], target[keep]), value[keep], accumulate=True)

    diagonal = kind == layer.DIAGONAL
    d = torch.ones(S, dtype=value.dtype).index_put((source[diagonal],), value[diagonal])
    slots = triangle(layer.LOWER) @ torch.diag(d) @ triangle(layer.UPPER)
    ins, outs = layer.input_slot, layer.output_slot
    w = slots.index_select(0, ins.clamp_min(0)).index_select(1, outs.clamp_min(0))
    return w * ((ins >= 0)[:, None] & (outs >= 0)[None, :]).to(w)
