@torch.no_grad()
def _update_contextual_concept_codebooks(self):
    """Commit the context-owned SBOW rotation for one completed sentence.

    The serial fullgraph has already consumed the staged concept rows by
    the time this runs.  We therefore take one detached snapshot of the
    same rows, form a leave-one-out bag-of-concepts context per word,
    contrast it against deterministic negative prototype rows, reduce all
    repeated targets by row, and make one exact sphere-rotation commit.

    No part of this method is called by the captured PS/WS/CS/SS word
    loop.  In particular it contains no scheduler/view-guard Python work
    in the compiled graph and no autograd edge to ``similarity_codebook``.
    The read/reduce/commit ordering means a row appearing several times in
    a batch always sees the same pre-update snapshot.
    """
    codebooks = tuple(getattr(
        self, "_contextual_concept_codebooks", ()) or ())
    rate = float(getattr(
        self, "contextual_concept_learning_rate", 0.0) or 0.0)
    if rate <= 0.0 or not codebooks:
        return None

    cb = codebooks[0]
    W = getattr(cb, "W", None)
    isp = getattr(self, "inputSpace", None)
    rows = getattr(isp, "_ar_word_concept_rows", None)
    active = getattr(isp, "_word_active_mask", None)
    if (not isinstance(cb, Codebook)
            or not bool(getattr(cb, "contextual_rotation_only", False))
            or not torch.is_tensor(W) or W.ndim != 2
            or not torch.is_tensor(rows) or rows.ndim != 2
            or not torch.is_tensor(active)
            or tuple(active.shape) != tuple(rows.shape)):
        return None

    active_rows = int(cb.active_row_count())
    if active_rows < 2:
        return None
    rows = rows.to(device=W.device, dtype=torch.long)
    active = active.to(device=W.device, dtype=torch.bool)
    valid = active & rows.ge(0) & rows.lt(active_rows)

    # A harmless in-range source is gathered for unknown/padded positions,
    # then immediately masked.  That keeps the contextual snapshot wholly
    # device-local and does not create a separate variable-width graph
    # inside the production word loop.
    safe_rows = rows.clamp(min=0, max=active_rows - 1)
    atoms = W[safe_rows]
    sentence_ids = getattr(isp, "_packed_sentence_ids", None)
    if (bool(getattr(isp, "_sentence_pack_enabled", False))
            and torch.is_tensor(sentence_ids)
            and tuple(sentence_ids.shape) == tuple(valid.shape)):
        sentence_ids = sentence_ids.to(
            device=W.device, dtype=torch.long)
        n_sentences = max(
            1, int(getattr(
                isp, "_packed_sentence_max_count", 1) or 1))
        safe_sentence = sentence_ids.clamp(
            min=0, max=n_sentences - 1)
        counts = torch.zeros(
            int(rows.shape[0]), n_sentences,
            dtype=atoms.dtype, device=W.device)
        counts.scatter_add_(
            1, safe_sentence,
            valid.to(dtype=atoms.dtype))
        sums = torch.zeros(
            int(rows.shape[0]), n_sentences, int(atoms.shape[-1]),
            dtype=atoms.dtype, device=W.device)
        sums.scatter_add_(
            1,
            safe_sentence.unsqueeze(-1).expand_as(atoms),
            atoms * valid.unsqueeze(-1).to(dtype=atoms.dtype))
        word_count = counts.gather(1, safe_sentence)
        valid = valid & word_count.gt(1)
        sentence_sum = sums.gather(
            1, safe_sentence.unsqueeze(-1).expand_as(atoms))
        denom = (word_count - 1.0).clamp_min(1.0)
        context = (
            sentence_sum - atoms * valid.unsqueeze(-1).to(atoms.dtype))
        context = context / denom.unsqueeze(-1)
    else:
        sentence_count = valid.sum(dim=1, keepdim=True)
        valid = valid & sentence_count.gt(1)
        atom_mask = valid.unsqueeze(-1).to(dtype=atoms.dtype)
        masked_atoms = atoms * atom_mask
        denom = (
            sentence_count.to(dtype=atoms.dtype) - 1.0).clamp_min(1.0)
        context = (
            masked_atoms.sum(dim=1, keepdim=True) - masked_atoms)
        context = context / denom.unsqueeze(-1)
    atom_mask = valid.unsqueeze(-1).to(dtype=atoms.dtype)
    # This normalizes only the *ephemeral context direction*.  It never
    # normalizes a percept input or re-projects a stored codebook row.
    context = F.normalize(context.float(), p=2, dim=-1, eps=1e-12)

    target = atoms.float()
    beta = 10.0
    positive_score = (target * context).sum(dim=-1, keepdim=True)
    tangent = (beta * torch.sigmoid(-beta * positive_score)) * context

    # Stateless integer hashing makes the negative pool deterministic for
    # a given staged sentence and training step.  The negative atoms are a
    # detached snapshot: only observed concept rows move, so one sentence
    # has one bounded owner commit rather than sparse optimizer writes to
    # arbitrary inventory rows.
    negative_count = int(getattr(
        self, "contextual_concept_negatives", 0) or 0)
    if negative_count > 0:
        B, width = (int(rows.shape[0]), int(rows.shape[1]))
        positions = torch.arange(
            width, device=W.device, dtype=torch.long).view(1, width, 1)
        offsets = torch.arange(
            1, negative_count + 1, device=W.device,
            dtype=torch.long).view(1, 1, negative_count)
        step = int(getattr(self, "_training_step_count", 0) or 0)
        hashed = (
            safe_rows.unsqueeze(-1) * 1103515245
            + positions * 2654435761
            + offsets * 2246822519
            + step * 3266489917
        )
        negative_rows = torch.remainder(hashed, active_rows)
        negative_rows = torch.where(
            negative_rows.eq(safe_rows.unsqueeze(-1)),
            torch.remainder(negative_rows + 1, active_rows),
            negative_rows)
        negative_atoms = W[negative_rows].float()
        negative_score = (target.unsqueeze(-2) * negative_atoms).sum(
            dim=-1, keepdim=True)
        negative_pull = (
            beta * torch.sigmoid(beta * negative_score) * negative_atoms
        ).mean(dim=-2)
        tangent = tangent - negative_pull

    # The exponential-map update consumes only tangential evidence.  A
    # word absent from its sentence context carries zero evidence and is
    # omitted before reduction, so unknown/padded row zero never receives
    # a structural write.
    tangent = tangent - (tangent * target).sum(
        dim=-1, keepdim=True) * target
    tangent = tangent * atom_mask.to(dtype=tangent.dtype)
    flat_valid = valid.reshape(-1)
    if not bool(flat_valid.any()):
        return None
    selected_rows = safe_rows.reshape(-1).masked_select(flat_valid)
    selected_tangent = tangent.reshape(-1, int(W.shape[1]))[flat_valid]
    unique_rows, inverse = torch.unique(
        selected_rows, sorted=True, return_inverse=True)
    reduced = torch.zeros(
        (int(unique_rows.shape[0]), int(W.shape[1])),
        device=W.device, dtype=selected_tangent.dtype)
    reduced.index_add_(0, inverse, selected_tangent)
    multiplicity = torch.zeros(
        (int(unique_rows.shape[0]), 1), device=W.device,
        dtype=selected_tangent.dtype)
    multiplicity.index_add_(
        0, inverse,
        torch.ones((int(inverse.shape[0]), 1), device=W.device,
                   dtype=selected_tangent.dtype))
    reduced = reduced / multiplicity.clamp_min(1.0)
    cb.rotate_rows(unique_rows, reduced, rate)
    return int(unique_rows.numel())
