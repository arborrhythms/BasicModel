"""Retired / parked classes, kept for easy revival.

These are NOT imported by the live model. The MATLAB-era ``Mem`` memory
primitives, ``DecisionBoundaryLayer``, and the retired ``QKVAttentionLayer``,
along with the documented-dormant grammar operators ``true`` / ``false`` /
``swap`` / ``copy`` / ``area`` / ``luminosity`` / ``isaPart``, were moved out
of ``bin/Layers.py`` in the 2026-07-17 legacy-code cleanup.

Revive by moving a class back into ``bin/Layers.py`` (and, for the grammar
operators, re-registering it in ``GRAMMAR_LAYER_CLASSES`` in
``bin/Language.py``). Tests exercise these here so revival stays cheap.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from util import TheXMLConfig, TheDevice

from Layers import (
    GrammarLayer,
    Layer,
    LinearLayer,
    Ops,
    area_op,
    isa_part_op,
    luminosity_op,
)


class TrueLayer(GrammarLayer):
    """``S -> true(S)`` -- keep only the pos pole of the bivector
    activation; zero the neg pole.

    The dispatcher hands TrueLayer the muxed event tensor
    (``reads_activation=False``); the bivector ``[pos, neg]`` lives
    at ``[..., :2]`` (same convention as ``NotLayer`` / ``NonLayer``).
    Returns the same shape with ``neg`` replaced by zero -- the
    bivector now affirms only what was on the positive pole, with
    no negative evidence. nWhere / nWhen channels at ``[..., 2:]``
    pass through unchanged.

    Lossy: the neg pole is destroyed; reverse is a passthrough.
    """
    rule_name        = "true"
    arity            = 1
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        """Initialize TrueLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        pos    = x[..., 0:1]
        zeros  = torch.zeros_like(pos)
        rest   = x[..., 2:]
        bivec  = torch.cat([pos, zeros], dim=-1)
        if rest.shape[-1] == 0:
            return bivec
        return torch.cat([bivec, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return y


class FalseLayer(GrammarLayer):
    """``S -> false(S)`` -- keep only the neg pole of the bivector
    activation; zero the pos pole.

    Mirror of ``TrueLayer``: dispatcher hands the muxed event tensor
    (``reads_activation=False``); bivector ``[pos, neg]`` at
    ``[..., :2]`` becomes ``[0, neg]``. nWhere / nWhen channels at
    ``[..., 2:]`` pass through unchanged.

    Lossy: the pos pole is destroyed; reverse is a passthrough.
    """
    rule_name        = "false"
    arity            = 1
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        """Initialize FalseLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        self._check_bivector_shape(x)
        neg    = x[..., 1:2]
        zeros  = torch.zeros_like(neg)
        rest   = x[..., 2:]
        bivec  = torch.cat([zeros, neg], dim=-1)
        if rest.shape[-1] == 0:
            return bivec
        return torch.cat([bivec, rest], dim=-1)

    def reverse(self, y):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return y


class SwapLayer(GrammarLayer):
    """``S -> swap(S, S)`` -- swap the left and right arguments.

    Forward returns ``right`` -- the right operand takes the
    canonical (parent) slot, simulating a left/right argument
    swap for downstream rules. Lossy (the original left is
    discarded); reverse is the symmetric ``(parent, parent)``
    pseudo-inverse.

    Parameter-free; no Sinkhorn / marker machinery (those were
    retired with the 2026-05-04 operator overhaul -- soft
    permutation belongs in the chart's CKY pair-selection logic,
    not in a per-cell GrammarLayer).
    """
    rule_name        = "swap"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        """Initialize SwapLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return right

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class CopyLayer(GrammarLayer):
    """``S -> copy(S, S)`` -- copy the left argument to the parent slot.

    Phase 1b dual of `SwapLayer`: forward returns ``left`` -- the
    left operand takes the canonical (parent) slot, discarding the
    right operand.  Lossy (the original right is unrecoverable);
    reverse is the symmetric ``(parent, parent)`` pseudo-inverse
    shared with `SwapLayer`.

    Parameter-free.  Like `swap`, the gradient signal that trains
    `copy` flows through ``Grammar.rule_probability('copy')`` and
    the chart's CKY pair-selection state — the standard mechanism
    already differentiable under the prediction loss.

    Naming and arity dual to `swap`:
        swap.forward(a, b) -> b      copy.forward(a, b) -> a
        swap.reverse(p)    -> (p, p) copy.reverse(p)    -> (p, p)
    """
    rule_name        = "copy"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        """Initialize CopyLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(0, 0)

    def forward(self, left, right):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        return left

    def reverse(self, parent):
        """Reverse pass; inverse of ``forward``.
        
        See class docstring for the inversion contract.
        """
        return parent, parent

    def compose(self, left, right):
        """Compose the input via this layer's parse contract."""
        return self.forward(left, right)

    def generate(self, parent):
        """Drive the reverse / generation pass."""
        return self.reverse(parent)


class AreaLayer(GrammarLayer):
    """``S -> area(S)`` -- introspective scalar in [0, 1]."""
    rule_name        = "area"
    arity            = 1
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, x):
        return area_op(x)

    def reverse(self, parent):
        return parent

    def compose(self, x):
        return self.forward(x)

    def generate(self, parent):
        return self.reverse(parent)


class LuminosityLayer(GrammarLayer):
    """``S -> luminosity(S, S)`` -- introspective scalar in [-1, 1]."""
    rule_name        = "luminosity"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'CS'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, left, right):
        return luminosity_op(left, right)

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)


class IsaPartLayer(GrammarLayer):
    """``S -> isaPart(S, S)`` -- one-step kernel overlap ∈ (0, 1]."""
    rule_name        = "isaPart"
    arity            = 2
    invertible       = False
    lossy            = True
    space_role             = 'SS'
    reads_activation = False

    def __init__(self):
        super().__init__(0, 0)

    def forward(self, child, parent):
        return isa_part_op(child, parent)

    def reverse(self, parent):
        return parent, parent

    def compose(self, left, right):
        return self.forward(left, right)

    def generate(self, parent):
        return self.reverse(parent)


class DecisionBoundaryLayer(Layer):
    """Learns a hyperplane normal vector via online updates (not backprop).

    forward() returns +1/-1 on each side of the boundary.
    update() nudges the weight toward or away from an observation depending
    on which side it falls.
    """
    def __init__(self, nInput, nOutput, learning_rate=0.01):
        """Initialize DecisionBoundaryLayer; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super(DecisionBoundaryLayer, self).__init__(nInput, nOutput)
        self.learning_rate = learning_rate
        self.weight        = nn.Parameter(torch.zeros(nInput, nOutput))
        self.register_buffer('noise', torch.randn(nInput, nOutput))

    def forward(self, x, t=0):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        if t != 0:
            self.noise = torch.randn(
                self.weight.shape,
                device=TheDevice.get(),
                dtype=self.weight.dtype,
            )

        W = self.weight + t*self.noise
        dot_product = torch.matmul(x, W)
        decision = torch.where(dot_product >= 0, torch.tensor(1.0, device=x.device),
                               torch.tensor(-1.0, device=x.device))
        return decision

    def update(self, x, t=0):
        """Update.
        
        See class docstring for the operation contract.
        """
        d1 = torch.norm(x - self.weight) ** 2
        d2 = torch.norm(x + self.weight) ** 2
        if d1 < d2:
            self.weight.data += self.learning_rate * (x.unsqueeze(1) - self.weight.data)
        else:
            self.weight.data -= self.learning_rate * (x.unsqueeze(1) + self.weight.data)

    @staticmethod
    def test():
        """Self-test; verifies the round-trip / invariant."""
        import matplotlib.pyplot as plt
        n_points = 100
        data = torch.randn(n_points, 2, device=TheDevice.get())
        data[:, 0] *= 1.5

        layer = DecisionBoundaryLayer(nInput=2, nOutput=1, learning_rate=0.01)
        for _ in range(1000):
            idx = torch.randint(0, n_points, (1,))
            x = data[idx].squeeze()
            layer.update(x)

        w = layer.weight.detach().cpu().numpy()
        w_neg = -w

        data_np = data.cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(data_np[:, 0], data_np[:, 1], label="Data", alpha=0.6)

        plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='r', linestyle='-',label='w')
        plt.quiver(0, 0, w_neg[0], w_neg[1], angles='xy', scale_units='xy', scale=1, color='b', linestyle='-',label='-w')

        a, b = w
        x_vals = np.linspace(np.min(data_np[:, 0]) - 1, np.max(data_np[:, 0]) + 1, 100)
        if np.abs(b) > 1e-5:
            y_vals = - (a / b) * x_vals
            plt.plot(x_vals, y_vals, color='g', linestyle='dashed', label='Hyperplane')
        else:
            plt.axvline(0, color='g', linestyle='dashed', label='Hyperplane')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Decision Boundary Learning')
        plt.legend()
        plt.grid(True)
        plt.xlim(np.min(data_np[:, 0]) - 2, np.max(data_np[:, 0]) + 2)
        plt.ylim(np.min(data_np[:, 1]) - 2, np.max(data_np[:, 1]) + 2)
        warnings.filterwarnings('ignore', message='.*line style')
        plt.show(block=False)


class QKVAttentionLayer(Layer):
    """Unified attention layer with three modes. Retired from enlistment
    (plan 2026-06-06-symbolic-heat-retrieval); kept for back-compat.

    type="symmetric"   -- Hopfield-like: scores = A^T @ A (positive semi-definite).
                         Attends across feature channels.
    type="asymmetric"  -- Channel attention: scores = Q^T @ K.
                         Attends across feature channels.
    type="transformer" -- Standard multi-head attention over the object/token axis.
                         Q K^T / sqrt(d) with multi-head splitting.

    All modes require 3D input [batch, nIdeas, dim].
    """
    def __init__(self, nInput, nOutput, nHidden=None, type="asymmetric", nHeads=1):
        """Initialize QKVAttentionLayer; allocate state for the class contract.

        See class docstring for invariants.
        """
        super(QKVAttentionLayer, self).__init__(nInput, nOutput)
        self.nHidden = nOutput if not nHidden else nHidden
        self.type = type
        self.mask = None
        self.beta = 10
        self.reversible = False
        self.nHeads = nHeads

        if self.type == "symmetric":
            self.A = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        else:
            self.Q = LinearLayer(self.nInput, self.nHidden)
            self.K = LinearLayer(self.nInput, self.nHidden)
            self.V = LinearLayer(self.nInput, self.nHidden)
        self.Out = LinearLayer(self.nHidden, self.nOutput)

        # Transformer-specific: multi-head geometry
        if self.type == "transformer":
            if nHeads < 1:
                raise ValueError(f"nHeads must be >= 1, got {nHeads}")
            if self.nHidden % self.nHeads != 0:
                raise ValueError(
                    f"nHidden ({self.nHidden}) must be divisible by nHeads ({self.nHeads})")
            self.headDim = self.nHidden // self.nHeads
            self.scale = self.headDim ** -0.5

    # --- Transformer helpers (multi-head) ---

    def _reshape_heads(self, x):
        batch, n_obj, _ = x.shape
        x = x.view(batch, n_obj, self.nHeads, self.headDim)
        return x.transpose(1, 2)

    def _normalize_mask(self, mask, batch, n_obj):
        """Normalize mask.
        
        See class docstring for the operation contract.
        """
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)
        if mask.dim() == 2:
            if list(mask.shape) != [batch, n_obj]:
                raise ValueError(
                    f"2D mask must have shape {[batch, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, None, :].expand(-1, self.nHeads, n_obj, -1)
        elif mask.dim() == 3:
            if list(mask.shape) != [batch, n_obj, n_obj]:
                raise ValueError(
                    f"3D mask must have shape {[batch, n_obj, n_obj]}, got {list(mask.shape)}")
            mask = mask[:, None, :, :].expand(-1, self.nHeads, -1, -1)
        elif mask.dim() == 4:
            if mask.shape[0] != batch or mask.shape[-2:] != (n_obj, n_obj):
                raise ValueError(
                    f"4D mask must end with {[n_obj, n_obj]} and batch {batch}, "
                    f"got {list(mask.shape)}")
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.nHeads, -1, -1)
            elif mask.shape[1] != self.nHeads:
                raise ValueError(
                    f"4D mask head dimension must be 1 or nHeads ({self.nHeads}), "
                    f"got {mask.shape[1]}")
        else:
            raise ValueError(f"Unsupported mask rank {mask.dim()}; expected 2, 3, or 4")
        return mask

    # --- Forward dispatch ---

    def forward(self, x):
        """Forward pass.
        
        See class docstring for the operation this layer applies.
        """
        assert x.ndim == 3, f"QKVAttentionLayer expects 3D input [B, N, D], got {list(x.shape)}"
        if self.type == "transformer":
            return self._forward_transformer(x)
        elif self.type == "symmetric":
            return self._forward_symmetric(x)
        else:
            return self._forward_asymmetric(x)

    def _forward_symmetric(self, x):
        """Forward symmetric.
        
        See class docstring for the operation contract.
        """
        a2     = self.A(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(a2.transpose(-2, -1), a2) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_asymmetric(self, x):
        """Forward asymmetric.
        
        See class docstring for the operation contract.
        """
        query  = self.Q(x)
        key    = self.K(x)
        value  = x if self.nHidden == self.nInput else self.V(x)
        scores = torch.matmul(query.transpose(-2, -1), key) / (self.nInput ** 0.5)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        attn = F.softmax(self.beta * scores, dim=-1) if not self.reversible else scores
        output = value @ attn
        if self.nHidden != self.nOutput:
            output = self.Out(output)
        return output

    def _forward_transformer(self, x):
        """Forward transformer.
        
        See class docstring for the operation contract.
        """
        batch, n_obj, _ = x.shape
        query = self._reshape_heads(self.Q(x))
        key   = self._reshape_heads(self.K(x))
        value = self._reshape_heads(self.V(x))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = self._normalize_mask(self.mask, batch, n_obj)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Cached for diagnostic readers (e.g. InterSentenceLayer uses
        # the last-position entropy as a confidence signal).  Detached
        # so holding it doesn't pin the graph.
        self.last_attn = attn.detach()
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch, n_obj, self.nHidden)
        output = self.Out(output)
        return output

    def reverse(self, y, bias=None, temp=None):
        """Attention is not analytically invertible; keep reverse as identity."""
        return super().reverse(y, bias=bias, temp=temp)

    @staticmethod
    def test():
        # Test all three types with 3D input
        """Self-test; verifies the round-trip / invariant."""
        for atype in ["symmetric", "asymmetric", "transformer"]:
            kwargs = {"nInput": 8, "nOutput": 8, "type": atype}
            if atype == "transformer":
                kwargs["nHeads"] = 2
            layer = QKVAttentionLayer(**kwargs)
            x = torch.randn(4, 5, 8, device=TheDevice.get())
            y = layer(x)
            assert list(y.shape) == [4, 5, 8], f"type={atype}: expected [4,5,8], got {list(y.shape)}"
        # Test nInput != nOutput
        layer = QKVAttentionLayer(nInput=6, nOutput=3, nHidden=7, type="asymmetric")
        x = torch.randn(4, 5, 6, device=TheDevice.get())
        y = layer(x)
        assert list(y.shape) == [4, 5, 3], f"asymmetric nIn!=nOut: expected [4,5,3], got {list(y.shape)}"


class Mem:
    """Base class for temporal memory filters (exponential, gamma, mean, etc.).

    Subclasses implement ``delta()`` to update internal state from a new
    observation.  ``get()`` returns the current filtered output.  The
    ``removeRC``/``insertRC``/``setRC`` helpers support dynamic resizing
    of the output matrix (1-indexed for legacy compatibility).
    """
    def __init__(self, sz=None):
        """Initialize Mem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        self.lr = 0.01
        self.nTrials = 0
        self.output = None
        self.reset(sz)

    def delta(self, *args):
        """Increment nTrials by one. Extra arguments are ignored."""
        self.nTrials += 1
    def get(self):
        """Return the current output array."""
        return self.output
    def set(self, in_val):
        """Set the output array to a new value."""
        self.output = in_val
    def reset(self, sz=None):
        """
        Reset the output to a zeros array and nTrials to 0.
        If sz is not provided, use the current output's shape (or (0, 0) if undefined).
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        self.output = torch.zeros(sz, device=TheDevice.get())
        self.nTrials = 0
    def removeRC(self, r=None, c=None):
        """
        Remove a row and/or column from the output.
        The row (r) and column (c) indices are assumed to be given in MATLAB 1-indexed form.
        """
        if r is not None:
            self.output = np.delete(self.output, r - 1, axis=0)
        if c is not None:
            self.output = np.delete(self.output, c - 1, axis=1)
    def insertRC(self, r=None, c=None):
        """
        Insert a row and/or column of zeros into the output.
        The indices r and c are assumed to be 1-indexed.
        """
        if r is not None:
            new_row = np.zeros((1, self.output.shape[1]))
            self.output = np.insert(self.output, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.output.shape[0], 1))
            self.output = np.insert(self.output, c - 1, new_col, axis=1)
    def setRC(self, r, c, val):
        """
        Set the (r, c) element to val and its symmetric element (c, r) to val.
        Indices are assumed to be 1-indexed.
        """
        self.output[r - 1, c - 1] = val
        self.output[c - 1, r - 1] = val

    @staticmethod
    def test():
        """
        Test method that creates instances of several Mem-derived classes
        and calls their testImpulse method.
        """
        names = ['ExponentialMem', 'GammaMem', 'MeanMem']
        for name in names:
            # Create an instance using globals() (similar to MATLAB's feval)
            m = globals()[name]()
            Mem.testImpulse(m, name)

    @staticmethod
    def testImpulse(m, name):
        """
        Test impulse response by incrementing and plotting the output.
        """
        import matplotlib.pyplot as plt
        plt.figure(1)
        x = np.arange(1, 1001)
        y = np.zeros(1000)
        m.delta(1.0)
        for i in range(len(x)):
            y[i] = m.get()
            m.delta(0)
        plt.plot(x, y)
        plt.title(f"Impulse Response: {name}")
        plt.show(block=False)


class ZOHMem(Mem):
    """Zero-Order Hold memory: ``output`` is whatever was last written.

    Simplest memory primitive in the family -- every ``delta(in1)``
    call overwrites ``self.output`` with ``in1``, ignoring trial count.
    Useful as a baseline / control in memory comparisons.
    """
    def __init__(self, sz=1):
        """Initialize ZOHMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2=None):
        # Call the base class delta and then set output to in1.
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = in1


class StateMem(Mem):
    """Memory primitive with a parallel ``state`` tensor alongside ``output``.

    Subclasses that need a recurrent / hidden state (RLSMem, GammaMem)
    inherit from this. ``reset`` / ``removeRC`` / ``insertRC`` /
    ``setRC`` all mirror their changes into the state tensor so it
    stays shape-aligned with output.
    """
    def __init__(self, sz=1):
        """Initialize StateMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.state = None
        self.reset(sz)

    def reset(self, sz=None):
        """
        Reset output and state to zeros and nTrials to 0.
        """
        if sz is None:
            if self.output is None:
                sz = (0, 0)
            else:
                sz = self.output.shape
        super().reset(sz)
        self.state = torch.zeros(sz, device=TheDevice.get())

    def delta(self, *args):
        # Just call the base class delta.
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta(*args)

    def removeRC(self, r=None, c=None):
        """
        Remove a row/column from both output and state.
        """
        super().removeRC(r, c)
        if r is not None:
            self.state = np.delete(self.state, r - 1, axis=0)
        if c is not None:
            self.state = np.delete(self.state, c - 1, axis=1)

    def insertRC(self, r=None, c=None):
        """
        Insert a row/column of zeros into both output and state.
        """
        super().insertRC(r, c)
        if r is not None:
            new_row = np.zeros((1, self.state.shape[1]))
            self.state = np.insert(self.state, r - 1, new_row, axis=0)
        if c is not None:
            new_col = np.zeros((self.state.shape[0], 1))
            self.state = np.insert(self.state, c - 1, new_col, axis=1)

    def setRC(self, r, c, val):
        """
        Set the (r,c) element in both output and state (and symmetrically).
        """
        super().setRC(r, c, val)
        self.state[r - 1, c - 1] = val
        self.state[c - 1, r - 1] = val


class RLSMem(StateMem):
    """Recursive least-squares memory: ``output += L2(output - in1) * in1``.

    Adaptive update keyed off the residual norm; large errors move the
    output toward ``in1`` more aggressively. ``momLR`` is the optional
    momentum learning rate for the (currently unused) state update path.
    """
    def __init__(self, sz=1):
        """Initialize RLSMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.momLR = 0.2

    def delta(self, in1, in2=None):
        # Call the base class (Mem) delta.
        """Delta.
        
        See class docstring for the operation contract.
        """
        Mem.delta(self)
        # Compute error using the L2 norm.
        err = np.linalg.norm(self.output - in1, 2)
        self.output = self.output + err * in1


class ProbMem(Mem):
    """Conditional-probability memory: each cell tracks P(in1_r | in2_c).

    On each ``delta(in1, in2)`` step, the per-cell value is updated as a
    running-mean toward +1 / -1 / 0 depending on the sign agreement of
    the corresponding inputs. Encodes positive vs negative co-occurrence.
    """
    def __init__(self, sz=1):
        """Initialize ProbMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        # Iterate over the indices of in1 and in2.
        for r in range(len(in1)):
            for c in range(len(in2)):
                # Increase or decrease conditional probability based on activation sign.
                if in2[c] > 0 and in1[r] > 0:
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * 1
                elif in2[c] > 0 and in1[r] < 0:
                    self.output[r, c] = ((self.nTrials - 1) / self.nTrials) * self.output[r, c] + (1 / self.nTrials) * -1


class MeanMem(Mem):
    """Running arithmetic mean of all inputs seen so far.

    ``delta(in1)`` updates ``output`` toward ``in1`` with weight
    ``1 / nTrials``, so the result is the exact unweighted mean of the
    sequence. Bias-free; new samples have diminishing influence.
    """
    def __init__(self, sz=1):
        """Initialize MeanMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2=None):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = ((self.nTrials - 1) / self.nTrials) * self.output + (1 / self.nTrials) * in1


class GammaMem(StateMem):
    """Two-stage exponential filter: state then output.

    ``state`` integrates ``in1`` at rate ``self.lr``; ``output``
    integrates ``state`` at rate ``self.lr2``. Effectively a cascaded
    low-pass with separately tunable bandwidths.
    """
    def __init__(self, sz=1, lr2=0.05):
        """Initialize GammaMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        self.lr2 = lr2

    def delta(self, in1, in2=None):
        # Call the StateMem delta method.
        """Delta.
        
        See class docstring for the operation contract.
        """
        StateMem.delta(self)
        self.state = (1 - self.lr) * self.state + self.lr * in1
        self.output = (1 - self.lr2) * self.output + self.lr2 * self.state


class ExponentialMem(Mem):
    """Exponential-moving-average memory: ``output = (1-lr)*output + lr*in1``.

    Single-rate IIR low-pass; equivalent to MeanMem at lr=1/n but with
    a fixed lr that gives exponentially decaying weight on old samples.
    Default lr inherits from the base ``Mem`` class.
    """
    def __init__(self, sz=1, lr=None):
        """Initialize ExponentialMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)
        if lr is not None:
            self.lr = lr

    def delta(self, in1, in2=None):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        self.output = (1 - self.lr) * self.output + self.lr * in1


class CorrMem(Mem):
    """Correlation matrix memory; tracks normalized product of two streams.

    On each ``delta(in1, in2)`` step, per-cell ``output[r, c]`` is
    updated toward ``saturate(in1[r]*in2[c] / (|in1[r]|*|in2[c]|))``
    with weight ``max(|in1[r]|, |in2[c]|) / nTrials`` -- favours
    high-amplitude updates.
    """
    def __init__(self, sz=1):
        """Initialize CorrMem; allocate state for the class contract.
        
        See class docstring for invariants.
        """
        super().__init__(sz)

    def delta(self, in1, in2):
        """Delta.
        
        See class docstring for the operation contract.
        """
        super().delta()
        for r in range(len(in1)):
            for c in range(len(in2)):
                val = in1[r] * in2[c]
                # Avoid division by zero.
                denom = np.sqrt(in1[r]**2 * in2[c]**2)
                if denom != 0:
                    val = Ops.saturate(val / denom)
                else:
                    val = 0
                amt = max(abs(in1[r]), abs(in2[c]))
                self.output[r, c] = ((self.nTrials - amt) / self.nTrials) * self.output[r, c] + (amt / self.nTrials) * val


# Surface elision policy for the parked binary ops (T5 BINARY_ELISION):
# copy keeps the survivor (order id), swap keeps the survivor with order
# swapped. Assigned here (rather than via Language's
# _OPERATOR_SURFACE_SCHEMAS loop) now that these classes are parked out of
# the live GRAMMAR_LAYER_CLASSES registry.
from Layers import T5_BINARY_ELISION  # noqa: E402
CopyLayer.surface_schema = T5_BINARY_ELISION
SwapLayer.surface_schema = T5_BINARY_ELISION
