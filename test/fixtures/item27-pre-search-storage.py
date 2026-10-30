"""Historical candidate search before the 2026-10-29 storage repair.

Preserved for value, choice, gradient and optimizer parity; never a production path.
"""
import torch

def _bounded_binary_reconstruction(op, parent, reference, on_right, known,
                                   basis, basis_valid, candidate_limit, *,
                                   left_valid=None, right_valid=None,
                                   left_priming=None, right_priming=None,
                                   left_basis=None, right_basis=None,
                                   chooser=None, return_details=False, require_progress=False, whole_bank=None, case_bank=None):
    """Reconstruction-owned hard pair, initially least residual; no straight-through search gradient.

    At most K constituents per side, with words snapped only at leaves.
    The search has
    at most K squared pairs, K=candidate_limit. Both evidence lanes are
    read independently only after the form has identified the words.
    Retrieval indices and any supplied reference are detached. Dictionary
    candidates are detached constants. Residuals are relative to the
    parent's mean square. The trial comparison trains the compose chooser.
    This is an approximate reconstruction; recomposition and child fidelity
    must be measured separately. No candidate means unavailable, never an
    identity pseudo-inverse. Callers own the snapshot and its row masks.
    """
    B, D = parent.shape
    absent = torch.zeros(B, dtype=torch.bool, device=parent.device)
    if basis is None or basis_valid is None or int(basis.shape[1]) == 0:
        return torch.zeros_like(parent), torch.zeros_like(parent), absent
    K = min(max(1, int(candidate_limit)), int(basis.shape[1]))
    from SentenceUnderstanding import readback_scores
    layout = getattr(op, 'meaning_layout', None)
    form_width = layout[0] if layout is not None and layout[1] == D else None
    def shortlist(valid, priming, candidates):
        candidates = (basis if candidates is None else candidates).detach()
        valid = basis_valid if valid is None else basis_valid & valid
        priming = torch.ones_like(valid, dtype=parent.dtype) if priming is None else priming
        scores = readback_scores(parent.detach(), candidates, priming,
                                 meaning_start=form_width)
        indices = scores.masked_fill(~valid, -torch.inf).topk(K, dim=-1).indices
        # Top-k chooses the search set. The pair chooser owns orientation.
        indices = indices.sort(dim=-1).values
        candidates = candidates.gather(1, indices[:, :, None].expand(B, K, D))
        active = valid.gather(1, indices)
        return torch.where(active[:, :, None], candidates, 0.), active, indices
    left_candidates, left_active, left_indices = shortlist(left_valid, left_priming, left_basis)
    right_candidates, right_active, right_indices = shortlist(right_valid, right_priming, right_basis)
    K = left_candidates.shape[1]
    older = left_candidates[:, :, None, :].expand(B, K, K, D)
    newer = right_candidates[:, None, :, :].expand(B, K, K, D)
    reference = reference.detach()
    older = torch.where((known & ~on_right)[:, None, None, None],
                        reference[:, None, None, :], older)
    newer = torch.where((known & on_right)[:, None, None, None],
                        reference[:, None, None, :], newer)
    allowed = left_active[:, :, None] & right_active[:, None, :]
    if require_progress:
        form = parent if form_width is None else parent[..., :form_width]
        lf = older if form_width is None else older[..., :form_width]
        rf = newer if form_width is None else newer[..., :form_width]
        tolerance = 1e-8 * form.square().mean(-1).clamp_min(1. / form.shape[-1])
        progressing = (((lf - form[:, None, None]).square().mean(-1) > tolerance[:, None, None])
                      & ((rf - form[:, None, None]).square().mean(-1) > tolerance[:, None, None]))
        # Reject self-reproducing pairs before ranking, so a live whole
        # cannot hide a supported lexical split with the same residual.
        allowed = allowed & progressing
    counts, whole = None, None
    if form_width is not None:
        # A symbol bank is signless. Solve each candidate's two evidence
        # lanes before judging an exact logical fold, as the selected
        # inverse has always done. Its stored stance is not this reading.
        from MeaningCodes import recover_lanes
        recovered_left, recovered_right = recover_lanes(getattr(op, 'meaning_rule', None),
            parent[:, None, None], older, newer, form_width)
        older = torch.where((known & ~on_right)[:, None, None, None], older, recovered_left)
        newer = torch.where((known & on_right)[:, None, None, None], newer, recovered_right)
        domain = ((older >= 0) & (older <= 1) & (newer >= 0) & (newer <= 1)).all(-1)
        allowed = allowed & domain
        older, newer = (torch.where(domain[..., None], value, 0.) for value in (older, newer))
        # Free generation proposes two contents, not a saved tree. Fit the
        # centroid mixture from their forms; this is the count ratio.
        from FormCentroids import fit
        ratio, whole = fit(parent[:, None, None, :form_width], older, newer, whole_bank)
        # A child occupies at least one leaf; endpoints are not splits.
        allowed = allowed & (ratio > 0) & (ratio < 1)
        counts = (ratio, 1-ratio)
    case_view = (case_bank.for_values(newer, form_width=form_width)
                 if case_bank is not None and form_width is not None and
                 callable(getattr(op, 'compose_case_meaning', None)) else None)
    folded = (op.compose_blocks(op.compose, older, newer, counts=counts, whole=whole, case_view=case_view) if form_width is not None else
              older if getattr(op, 'reconstructs_left', False) else op.compose(older, newer))
    if form_width is None and getattr(op, 'same_reference_idempotent', False):
        same = left_indices[:, :, None] == right_indices[:, None, :]
        # A known occurrence is not identified by an unrelated basis row.
        same = same & ~known[:, None, None] & (older == newer).all(-1)
        from MeaningCodes import repeated_symbol
        folded = torch.where(same[..., None], repeated_symbol(older, form_width), folded)
    form = parent if form_width is None else parent[..., :form_width]
    folded_form = folded if form_width is None else folded[..., :form_width]
    residual = (folded_form - form[:, None, None, :]).square().mean(-1)
    energy = form.square().mean(-1)[:, None, None]
    # Exact zero origins use the existing Error squared-penalty convention;
    # no floor changes any positive parent scale.
    residual = residual / torch.where(energy > 0, energy, torch.ones_like(energy))
    if form_width is not None:
        meaning = parent[..., form_width:]
        meaning_energy = meaning.square().mean(-1)[:, None, None]
        residual = residual + (folded[..., form_width:] - meaning[:, None, None]).square().mean(-1) / torch.where(
            meaning_energy > 0, meaning_energy, 1.)
    # The shortlist and eligibility are unchanged. Context learns only
    # through the separate teacher loss, never through candidate codes.
    def candidate_features(candidates, indices, priming, active):
        candidates = candidates if form_width is None else candidates[..., :form_width]
        energy = candidates.square().sum(-1)
        activation = (form.detach()[:, None] * candidates).sum(-1) / torch.where(
            energy > 0, energy, torch.ones_like(energy))
        count = active.sum(-1, keepdim=True).clamp_min(1)
        mean = torch.where(active, activation, 0.).sum(-1, keepdim=True) / count
        centered = torch.where(active, activation - mean, 0.)
        variance = centered.square().sum(-1, keepdim=True) / count
        activation = centered / variance.sqrt().clamp_min(torch.finfo(activation.dtype).eps)
        heat = (torch.ones_like(activation) if priming is None else
                priming.detach().gather(1, indices))
        return activation, heat
    la, lp = candidate_features(left_candidates, left_indices, left_priming, left_active)
    ra, rp = candidate_features(right_candidates, right_indices, right_priming, right_active)
    features = torch.stack((-residual.detach(), la[:, :, None].expand_as(residual),
        ra[:, None, :].expand_as(residual), lp[:, :, None].expand_as(residual),
        rp[:, None, :].expand_as(residual)), -1)
    if chooser is None:
        selected = residual.masked_fill(~allowed, torch.inf).flatten(1).argmin(-1)
        logits = -residual.detach().masked_fill(~allowed, torch.inf).flatten(1)
        logits = torch.where(allowed.flatten(1).any(-1)[:, None], logits, 0.)
    else:
        selected, _, logits = chooser(features, allowed)
    # Any exact recomposition outranks an approximate one. Within that
    # tolerance the learned chooser decides, so roundoff cannot bypass
    # its orientation lesson. CE still sees every legal pair.
    fits = residual.detach().masked_fill(~allowed, torch.inf).flatten(1)
    exact_pairs = fits <= (8 * torch.finfo(parent.dtype).eps) ** 2
    exact = exact_pairs.any(-1)
    exact_choice = logits.masked_fill(~exact_pairs, -torch.inf).argmax(-1)
    selected = torch.where(exact, exact_choice, selected)
    gather = selected[:, None, None].expand(B, 1, D)
    hard_left = older.reshape(B, K * K, D).gather(1, gather).squeeze(1)
    hard_right = newer.reshape(B, K * K, D).gather(1, gather).squeeze(1)
    if form_width is not None:
        from MeaningCodes import recover_lanes
        hard_left, hard_right = recover_lanes(getattr(op, 'meaning_rule', None),
            parent, hard_left, hard_right, form_width)
        if (not callable(getattr(op, 'compose_case_meaning', None)) and
                (getattr(op, 'meaning_kernel', None) is not None or getattr(op, 'meaning_action', None) is not None)):
            fraction = ratio.flatten(1).gather(1, selected[:, None])[:, 0]
            right_known = op.meaning_head_role == 0 or getattr(op, 'meaning_action', None) is not None
            _, selected_whole = fit(parent[..., :form_width], hard_left, hard_right, whole_bank)
            left, right = op.reverse_blocks(parent, hard_right if right_known else hard_left,
                counts=(fraction, 1-fraction), reference_right=right_known, whole=selected_whole)
            # The inverse recovers meaning; the selected symbol supplies
            # its exact form. No parameter lives on the centroid side.
            left = torch.cat((hard_left[..., :form_width], left[..., form_width:]), -1)
            right = torch.cat((hard_right[..., :form_width], right[..., form_width:]), -1)
            domain = ((left >= 0) & (left <= 1) & (right >= 0) & (right <= 1)).all(-1)
            hard_left = torch.where(domain[:, None], left, hard_left)
            hard_right = torch.where(domain[:, None], right, hard_right)
    available = allowed.flatten(1).any(-1)
    hard_left = torch.where(available[:, None], hard_left, 0.)
    hard_right = torch.where(available[:, None], hard_right, 0.)
    result = (hard_left.detach(), hard_right.detach(), available) if form_width is None else (hard_left, hard_right, available)
    if return_details:
        return (*result, dict(logits=logits, selected=selected, allowed=allowed,
            left_indices=left_indices, right_indices=right_indices, features=features,
            exact=exact, relative_residual=residual.detach()))
    return result
