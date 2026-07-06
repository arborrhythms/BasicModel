"""Order-k membership unfold (snap contract sec 2.3, 2026-07-06): before the
peel's inner product, reconstitute each stored row to its pre-fold value via
``invert_ramsified`` so an order-k region is matched by its unfolded form,
not the folded order-0 shadow. Locks: (1) ``Codebook.unfolded_prototypes``
reconstitutes stamped rows and passes unstamped rows through; (2) the
silent-degradation guard (None without a ramsification table -> the caller
falls back to order-0); (3) the peel's ``prototypes=`` hook peels in the
supplied basis while indexing the codebook rows; (4) THE BAR -- with a
TRAINED (non-identity) fold, a pre-fold-domain member misses the folded row
at order 0 but is recovered exactly through the unfold.

Probe hygiene (Alec, 2026-07-06): the invertible LDU initializes at IDENTITY
(zero raw_L/raw_U), so a FRESH fold preserves direction trivially -- any
direction-change test must perturb/train the weights first; and ``set_sigma``
engages the ergodic path, whose first iteration is degenerate (constants).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
import Layers  # noqa: E402
from Language import ChunkLayer  # noqa: E402
from Spaces import Codebook  # noqa: E402


class _Shim:
    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


def _codebook(V, D):
    cb = Codebook()
    cb.create(D, V, D, customVQ=False, monotonic=False)
    return cb


def test_unfolded_prototypes_reconstitutes_stamped_rows():
    # The FOLD_SIGMA stamp means "this row WAS produced by the fold" -- so a
    # stamped row must hold a genuine fold OUTPUT (reversing an arbitrary
    # row through a trained fold saturates at the atanh rails, correctly:
    # it is out of the fold's range).
    D = 4
    sigma = _trained_sigma(D, seed=8)
    torch.manual_seed(9)
    member = (torch.randn(1, 1, D) * 0.6).tanh()
    cb = _codebook(V=5, D=D)
    with torch.no_grad():
        cb.getW()[2] = sigma.forward(member).reshape(-1)
    cb.enable_ramsification(max_order=1)
    cb.record_fold(torch.tensor([2]), 0, Codebook.FOLD_SIGMA)   # row 2 is order-1
    proto = cb.unfolded_prototypes(sigma=sigma)
    assert proto is not None and tuple(proto.shape) == (5, D)
    W = cb.getW()
    # Row 2 is reconstituted: back at the pre-fold member, and re-folds to W[2].
    assert torch.allclose(proto[2], member.reshape(-1), atol=1e-3)
    assert torch.allclose(sigma.forward(proto[2:3].reshape(1, 1, D)).reshape(-1),
                          W[2], atol=1e-3)
    # The trained fold genuinely moved the row (the unfold is not a no-op)...
    assert not torch.allclose(proto[2], W[2], atol=1e-2)
    # ...while unstamped rows (all NEITHER) pass through unchanged.
    for r in (0, 1, 3, 4):
        assert torch.allclose(proto[r], W[r], atol=1e-6), f"row {r} not identity"


def test_unfolded_prototypes_none_without_table():
    # Silent-degradation guard: no ramsification table -> None (the caller
    # peels against getW() directly, the order-0 fallback -- never errors).
    cb = _codebook(V=3, D=4)
    assert cb.unfolded_prototypes() is None


def test_peel_runs_in_the_prototypes_basis():
    # The peel selects rows by the SUPPLIED basis while its output still indexes
    # the codebook rows. Same query, two bases -> different selected row.
    D = 4
    e = torch.eye(D)
    W = torch.stack([e[0], e[1], e[2]])            # row r ~ axis r
    P = torch.stack([e[1], e[0], e[2]])            # rows 0,1 swapped
    query = e[0].clone()
    parts_w, _ = ChunkLayer.peel(query, _Shim(W), max_parts=1)
    parts_p, _ = ChunkLayer.peel(query, _Shim(W), prototypes=P, max_parts=1)
    assert parts_w[0][0] == 0, "order-0 basis: query e0 selects row 0"
    assert parts_p[0][0] == 1, "prototypes basis: e0 lives on P's row 1"


def _trained_sigma(D, seed=5, scale=0.5):
    """An invertible sigma fold with PERTURBED (trained-like) weights -- the
    identity-initialized LDU is trivially direction-preserving, so any
    order-k direction test must move the weights off identity first."""
    torch.manual_seed(seed)
    s = Layers.SigmaLayer(D, D, naive=True, invertible=True)
    with torch.no_grad():
        for _n, p in s.layer.named_parameters():
            p.add_(scale * torch.randn_like(p))
    return s


def test_order_k_member_missed_at_order0_recovered_by_unfold():
    # THE BAR (design sec 2.3 / T5 verify): a member expressed in the
    # PRE-FOLD domain misses the folded row at order 0 (the trained fold
    # rotated it -- the order-0 shadow), but the unfolded prototype matches
    # it exactly.
    D = 8
    sigma = _trained_sigma(D)
    torch.manual_seed(6)
    member = (torch.randn(1, 1, D) * 0.6).tanh()   # pre-fold-domain member
    folded = sigma.forward(member).reshape(-1)     # what the codebook stores
    cb = _codebook(V=4, D=D)
    with torch.no_grad():
        cb.getW()[2] = folded
    cb.enable_ramsification(max_order=1)
    cb.record_fold(torch.tensor([2]), 0, Codebook.FOLD_SIGMA)

    q = member.reshape(-1)
    cos0 = float(torch.nn.functional.cosine_similarity(
        q, folded, dim=0).abs())
    assert cos0 < 0.9, (
        f"premise: the trained fold must rotate direction (|cos|={cos0:.3f})")

    # Order-0 probe (no prototypes): the single-row fit leaves a large
    # residual -- the folded row is only a shadow of the member's direction.
    parts0, res0 = ChunkLayer.peel(q, cb, max_parts=1)
    assert float(res0.norm()) > 0.3 * float(q.norm()), (
        "order-0 match should visibly miss the pre-fold-domain member")

    # Unfolded probe: reconstitute the stamped row, peel in that basis --
    # the member is recovered as row 2 at coeff ~ 1 with residual ~ 0.
    proto = cb.unfolded_prototypes(sigma=sigma)
    parts, residual = ChunkLayer.peel(q, cb, prototypes=proto, max_parts=1)
    assert parts and parts[0][0] == 2, parts
    assert abs(parts[0][1] - 1.0) < 5e-3, parts
    assert float(residual.norm()) < 1e-3 * (1 + float(q.norm()))


def test_unfolded_prototypes_feed_the_peel():
    # End-to-end wiring: the unfolded matrix is an accepted prototypes= arg and
    # the peel recovers a query built in the unfolded basis.
    D = 4
    sigma = _trained_sigma(D, seed=7)
    cb = _codebook(V=4, D=D)
    cb.enable_ramsification(max_order=1)
    cb.record_fold(torch.tensor([1]), 0, Codebook.FOLD_SIGMA)
    proto = cb.unfolded_prototypes(sigma=sigma)
    query = proto[1].clone()                       # a member of row 1's region
    parts, residual = ChunkLayer.peel(query, cb, prototypes=proto, max_parts=1)
    assert parts and parts[0][0] == 1
    assert abs(parts[0][1] - 1.0) < 1e-3
    assert float(residual.norm()) < 1e-4 * (1 + float(query.norm()))
