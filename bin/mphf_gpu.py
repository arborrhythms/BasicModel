"""Static minimal-perfect-hash (MPHF) percept->row index for the
FROZEN-vocab training path -- the Rework A core mechanism.

Per the consolidated two-loop spec (§"Percept -> MPHF -> table",
§IMPLEMENTATION DETAILS D2): each percept's byte slot passes through an
MPHF producing an ``index in [0, V_percept)``. That index addresses a
**table** whose every entry holds BOTH (1) the literal surface word
string and (2) the ConceptualSpace activation vector for that token.

Crucial reuse (verified against bin/Spaces.py): the table's two halves
ALREADY EXIST on the frozen ``Embedding`` codebook -- they are NOT new
parallel structures:

  * the **concept-activation vector half** == ``codebook.wv._vectors``
    (the Phase-1A.1 learnable ``nn.Parameter`` lexicon rows -- gradient
    trained, REUSED here, never duplicated: a second embedding over the
    same surface tokens would double-count gradient);
  * the **surface-string half** == ``codebook.wv.index_to_key`` (already
    ASCII-prefilled by ``Embedding.create``: ``\\x00`` at row 0 / the
    NULL char + per-row cursor seal, ``chr(1)..chr(126)`` at the matching
    low rows, ``NULL_PERCEPT_KEY`` at ``codebook.null_percept_idx``; NO
    MASK row -- MASK is the all-zeros gaussian-tail effect, not a row).

So Rework A adds ONLY the static **MPHF index function** mapping a
percept byte slot -> the EXISTING ``key_to_index`` row, plus the
non-invertible reverse map (concept-activation vector -> nearest
``wv._vectors`` row -> ``index_to_key[row]``).

Build/verify pattern mirrors ``bin/bpe_gpu.py:build_static_tables`` /
``gpu_longest_match`` (frozen, built ONCE at the CPU->GPU handoff over
the frozen lexicon key set, cached; runtime is pure static tensor ops
-- poly-hash + ``searchsorted`` + collision-proof byte-verify -- ZERO
host sync, O(1) per slot). The MPHF is **non-invertible**: the reverse
map is the table lookup, never an inverse hash. A one-row silent
mis-resolution corrupts all training, so the byte-verify is exact
regardless of hash collisions (fail-loud memory).
"""
import torch

from bpe_gpu import _poly_hash  # reuse the verified FNV-style poly hash


class _MPHFUnavailable(Exception):
    """Raised when the static MPHF table cannot be built (e.g. an empty
    or non-Embedding codebook). The caller falls back to the existing
    verified path -- never a silent wrong result."""


def build_mphf_table(codebook, device):
    """Frozen lexicon key set -> static device tensors. Call ONCE per
    (frozen) codebook on the target device; cache the result.

    The frozen key set is ``codebook.wv.index_to_key`` (the ``.kv``
    lexicon + ASCII bootstrap + NULL rows -- frozen at
    ``word_learning<=0``). All Python-dict / utf-8 work happens HERE
    (one-time, host), never per batch.

    Returns a dict the runtime ``mphf_index`` consumes plus the
    ``surface`` list (== ``index_to_key``) for the reverse map.
    """
    wv = getattr(codebook, "wv", None)
    if wv is None or getattr(wv, "index_to_key", None) is None:
        raise _MPHFUnavailable(
            "build_mphf_table: codebook has no wv.index_to_key "
            "(non-Embedding / numeric codebook).")
    keys = list(wv.index_to_key)
    V = len(keys)
    if V == 0:
        raise _MPHFUnavailable("build_mphf_table: empty lexicon.")

    # Each key -> its utf-8 byte tuple, EXACTLY as InputSpace lexes a
    # token slot into ``subspace.what.W`` (null-terminated utf-8). The
    # MPHF keys on the same byte representation the percept slot carries
    # so the lookup is byte-for-byte the frozen ``key_to_index`` row.
    key_bytes = []
    maxL = 1
    for k in keys:
        try:
            kb = k.encode("utf-8")
        except Exception:
            # Non-str keys are not lexable percept slots; map to a
            # zero-length entry (never matched -> NULL row fallback).
            kb = b""
        key_bytes.append(kb)
        if len(kb) > maxL:
            maxL = len(kb)

    # Padded row->bytes (-1 pad) + lengths for the collision-proof
    # byte-verify, mirroring ``bpe_gpu``'s ``tok_bytes``/``tok_len``.
    tok_bytes = torch.full((V, maxL), -1, dtype=torch.int64)
    tok_len = torch.zeros(V, dtype=torch.int64)
    for i, kb in enumerate(key_bytes):
        kl = len(kb)
        tok_len[i] = kl
        if kl:
            tok_bytes[i, :kl] = torch.tensor(
                [int(b) for b in kb], dtype=torch.int64)

    # Per-length sorted (poly-hash -> row idx) index for the O(1)
    # exact-key ``searchsorted`` lookup. A length-0 key (empty / NULL
    # sentinel slot) is never matched -- the runtime maps an empty slot
    # to ``null_row`` explicitly, not via the hash.
    by_len = {}
    for L in range(1, maxL + 1):
        ids = [i for i in range(V) if int(tok_len[i]) == L]
        if not ids:
            by_len[L] = None
            continue
        ids_t = torch.tensor(ids, dtype=torch.int64)
        windows = tok_bytes[ids_t, :L]                 # [K, L]
        hashes = _poly_hash(windows)                   # [K]
        order = torch.argsort(hashes)
        by_len[L] = (hashes[order].to(device),
                     ids_t[order].to(device))

    # The NULL row: ``\x00`` (byte 0) is the per-row cursor seal / pad
    # sentinel and ``Embedding.create`` seeds it at row 0. An empty /
    # all-zero percept slot resolves here (the NULL char surface), NOT
    # via the hash.
    null_row = int(wv.key_to_index.get("\x00", 0))

    return {
        "maxL": int(maxL),
        "V": int(V),
        "tok_bytes": tok_bytes.to(device),
        "tok_len": tok_len.to(device),
        "by_len": by_len,
        "null_row": null_row,
        # The surface-string half of the D2 table == index_to_key
        # (ASCII-prefilled + NULL; the reverse map's lookup target).
        "surface": keys,
    }


def mphf_index(token_byte_slots, tables, return_verified=False):
    """``token_byte_slots`` [B, K, W] long (0..255; 0 terminates a slot,
    EXACTLY ``InputSpace.subspace.what.W``'s per-token null-terminated
    utf-8 layout).

    Returns ``row`` [B, K] long: the frozen lexicon row index for each
    percept slot. Pure static tensor ops (poly-hash + ``searchsorted``
    + collision-proof byte-verify), ZERO host sync, O(1) per slot.

    Resolution per slot:
      * effective key length L = #leading non-zero bytes (utf-8 token,
        null-terminated -- the byte AFTER the token is the 0 sentinel);
      * L == 0 (empty / pad / NULL-sentinel slot) -> ``null_row``;
      * else poly-hash the L-byte window, ``searchsorted`` the
        length-L sorted hash table, byte-verify against ``tok_bytes``
        (collision-proof: exact regardless of hash); a verified hit ->
        that row, a miss (key not in the frozen lexicon -- OOV) ->
        ``null_row`` (the documented frozen-vocab fallback: the table
        only holds the frozen key set; an OOV percept reconstructs to
        the NULL surface rather than silently aliasing a wrong row).

    When ``return_verified=True``, returns ``(row, verified)`` where
    ``verified`` ``[B, K]`` bool is True only for slots that hit the
    frozen lexicon via the collision-proof byte-verify (False for
    L==0 slots AND OOV slots; the row at those False positions is the
    ``null_row`` fallback). The ``chunking_mode='mphf'`` selectable
    runtime path uses this signal to gate the OOV->BPE-trie fallback
    per spec.
    """
    if token_byte_slots.dim() == 2:
        token_byte_slots = token_byte_slots.unsqueeze(-1)
    B, K, W = token_byte_slots.shape
    dev = token_byte_slots.device
    bb = token_byte_slots.to(torch.int64).clamp(0, 255)  # [B,K,W]
    null_row = tables["null_row"]
    maxL = tables["maxL"]

    # Effective key length = #leading non-zero bytes (the slot is one
    # token's utf-8 bytes then a 0). ``cumprod`` over (byte != 0) gives
    # 1 for every position up to (not including) the first 0; the sum
    # is the leading-run length -- pure tensor, no host sync.
    nonzero = (bb != 0).to(torch.int64)                  # [B,K,W]
    lead = torch.cumprod(nonzero, dim=-1)                # [B,K,W]
    eff_len = lead.sum(dim=-1)                            # [B,K]

    row = torch.full((B, K), null_row, dtype=torch.int64, device=dev)
    verified_any = torch.zeros((B, K), dtype=torch.bool, device=dev)
    for L in range(1, min(maxL, W) + 1):
        entry = tables["by_len"].get(L)
        if entry is None:
            continue
        keys_sorted, ids_sorted = entry
        win = bb[..., :L]                                 # [B,K,L]
        h = _poly_hash(win)                               # [B,K]
        pos = torch.searchsorted(keys_sorted, h)
        pos = pos.clamp(max=keys_sorted.numel() - 1)
        hit = keys_sorted[pos] == h
        cand = ids_sorted[pos]                            # [B,K]
        cb = tables["tok_bytes"][cand][..., :L]           # [B,K,L]
        verified = hit & (win == cb).all(dim=-1) & (eff_len == L)
        row = torch.where(verified, cand, row)
        verified_any = verified_any | verified
    if return_verified:
        return row, verified_any
    return row


def reverse_map_rows(concept_vectors, codebook):
    """Non-invertible reverse map: a concept-activation vector ->
    nearest ``wv._vectors`` row index.

    The MPHF is NOT inverted; the reverse map is the **table lookup**
    (nearest concept-activation row). ``concept_vectors`` is ``[..., D]``
    (D == ``wv._vectors`` width). Returns the matching ``[...]`` long
    row indices; ``surface[idx]`` (== ``codebook.wv.index_to_key[idx]``)
    is the reconstructed literal surface word. Pure tensor op, no host
    sync (the host ``index_to_key`` indexing is the caller's choice,
    done only off the training-critical path).
    """
    W = codebook.wv._vectors                              # [V, D]
    flat = concept_vectors.reshape(-1, concept_vectors.shape[-1])
    # Cosine-style nearest row (the codebook lives on the periodic unit
    # cell; dot-product nearest is the same neighbour the codebook
    # search uses). Static [N, V] then argmax -- no host sync.
    sims = flat @ W.t()                                   # [N, V]
    idx = sims.argmax(dim=-1)                              # [N]
    return idx.reshape(concept_vectors.shape[:-1])
