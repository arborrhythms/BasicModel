"""GPU BPE tokenizer for the FROZEN-vocab training path.

The legacy ``ChunkLayer`` greedy longest-match is a Python trie walk
that needs ``byte_indices.tolist()`` -- a per-step cudaMemcpyDtoH. When
the BPE vocab is frozen (``word_learning <= 0`` -- the CPU-pretrain ->
freeze -> GPU-train workflow), the whole tokenizer can be static tensor
ops with zero host sync:

  * ``build_static_tables`` -- ONCE at the CPU->GPU handoff, turn the
    frozen vocab into static device tensors: padded id->bytes, lengths,
    a per-length sorted (polyhash -> id) index for longest-match, a
    boundary-chunk mask (the word separators), and a chunk_id ->
    codebook_row map (collapses the per-call latin1/key_to_index dict
    chain).
  * ``gpu_longest_match`` -- per position, the longest vocab entry that
    is a prefix there. Fully parallel: for each length L, hash the
    L-byte window, ``searchsorted`` the sorted vocab hashes, then
    **byte-verify** against the padded id->bytes (collision-proof, so
    exact regardless of hash). ``_max_merge_len`` is 9 here (>8) so a
    single int64 pack will not fit -- hence hash + verify.
  * ``gpu_chunk_ids`` -- greedy left-to-right consumption. That step is
    inherently sequential (token k+1 starts at cursor+L[cursor]); done
    as a bounded on-device scan over a ``[B]`` cursor (no host sync).

Bit-identical to the trie walk is asserted by ``test/bpe_gpu_equiv.py``
before the switch -- a one-token silent divergence corrupts all
training (fail-loud memory).
"""
import torch


class _BPEGpuUnavailable(Exception):
    """Raised when the static GPU tables cannot be built (e.g. a frozen
    codebook/vocab key mismatch). The caller falls back to the verified
    trie reference -- never a silent wrong result."""


# Polynomial rolling hash over bytes. 1099511628211 is the FNV-style
# 64-bit prime; arithmetic wraps mod 2**64 via int64 overflow, which is
# fine -- the hash only needs to be a consistent bucket; correctness
# comes from the explicit byte-verify, not from the hash being perfect.
_HASH_MUL = 1099511628211


def _poly_hash(windows):
    """``windows`` [..., L] int64 byte values -> [...] int64 hash."""
    h = torch.zeros(windows.shape[:-1], dtype=torch.int64,
                    device=windows.device)
    L = windows.shape[-1]
    for k in range(L):
        h = h * _HASH_MUL + (windows[..., k] + 1)
    return h


def build_static_tables(chunk_layer, codebook, device):
    """Frozen vocab -> static device tensors. Call ONCE per (frozen)
    chunk_layer/codebook on the target device; cache the result.

    Returns a dict the GPU tokenizer + the rewired _embed_bpe consume.
    All Python-dict / latin1 work happens HERE (one-time, host), never
    per batch.
    """
    vocab = chunk_layer.vocab               # {byte-tuple: id}
    id_to_bytes = chunk_layer.id_to_bytes   # {id: byte-tuple}
    maxL = int(chunk_layer._max_merge_len)
    V = int(chunk_layer._next_id)           # ids are 0..V-1
    boundary = set(int(b) for b in chunk_layer.BOUNDARY_BYTES)

    tok_bytes = torch.full((V, maxL), -1, dtype=torch.int64)
    tok_len = torch.zeros(V, dtype=torch.int64)
    is_boundary = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        key = id_to_bytes.get(i)
        if key is None:
            # 0..255 are always seeded; a gap above that would be a
            # vocab bug -- single-byte fallback id == byte value.
            key = (i,) if i < 256 else ()
        kl = len(key)
        tok_len[i] = kl
        if kl:
            tok_bytes[i, :kl] = torch.tensor(
                [int(b) for b in key], dtype=torch.int64)
            is_boundary[i] = all(int(b) in boundary for b in key)

    # chunk_id -> codebook row (frozen resolve chain). -1 == "skip"
    # (boundary chunk, or byte_mode key not in the codebook -- the
    # original returns None and the sub-token is dropped).
    byte_mode = bool(getattr(codebook, 'byte_mode', False))
    key_to_index = codebook.pretrain.key_to_index
    chunk_to_cb = torch.full((V,), -1, dtype=torch.int64)
    for i in range(V):
        if bool(is_boundary[i]):
            continue
        key = id_to_bytes.get(i, (i,) if i < 256 else ())
        if not key:
            continue
        latin1 = "".join(chr(int(b) & 0xFF) for b in key)
        idx = key_to_index.get(latin1)
        if idx is None:
            if byte_mode:
                continue          # original: _resolve -> None -> skip
            raise AssertionError(
                f"build_static_tables: key {latin1!r} missing from "
                f"frozen codebook.pretrain (word_learning<=0) -- .kv "
                f"load mismatch.")
        chunk_to_cb[i] = int(idx)

    # Per-length sorted (hash -> id) for longest-match searchsorted.
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

    return {
        "maxL": maxL, "V": V,
        "tok_bytes": tok_bytes.to(device),
        "tok_len": tok_len.to(device),
        "is_boundary": is_boundary.to(device),
        "chunk_to_cb": chunk_to_cb.to(device),
        "by_len": by_len,
    }


def gpu_longest_match(byte_buf, tables):
    """``byte_buf`` [B, N] long (0..255; 0 terminates a row).

    Returns ``best_id`` [B, N] long and ``best_len`` [B, N] long: the
    id / length of the longest vocab entry starting at each position
    (single-byte fallback guarantees ``best_len >= 1``). Pure tensor
    ops, no host sync.
    """
    B, N = byte_buf.shape
    dev = byte_buf.device
    maxL = tables["maxL"]
    tok_bytes = tables["tok_bytes"]
    # Single-byte baseline: id == byte value (ids 0..255 seeded), len 1.
    best_id = byte_buf.clone()
    best_len = torch.ones(B, N, dtype=torch.int64, device=dev)

    for L in range(2, maxL + 1):
        entry = tables["by_len"].get(L)
        if entry is None or N < L:
            continue
        keys_sorted, ids_sorted = entry
        # [B, N-L+1, L] windows; pad-free positions only.
        win = byte_buf.unfold(1, L, 1).to(torch.int64)     # [B, M, L]
        h = _poly_hash(win)                                # [B, M]
        pos = torch.searchsorted(keys_sorted, h)
        pos = pos.clamp(max=keys_sorted.numel() - 1)
        hit = keys_sorted[pos] == h
        cand_id = ids_sorted[pos]                          # [B, M]
        # Byte-verify (collision-proof): window == tok_bytes[cand_id].
        cb = tok_bytes[cand_id][..., :L]                   # [B, M, L]
        verified = hit & (win == cb).all(dim=-1)           # [B, M]
        # A match of length L at position i beats any shorter one.
        M = win.shape[1]
        sl = slice(0, M)
        take = verified
        best_id[:, sl] = torch.where(take, cand_id, best_id[:, sl])
        best_len[:, sl] = torch.where(
            take, torch.full_like(best_len[:, sl], L),
            best_len[:, sl])
    return best_id, best_len


def segment_words(chunk_ids, tok_count, tables, nObj):
    """Tensor word-segmentation, **fully static** (no ``.item()``, no
    boolean-mask compaction -> zero DtoH, fullgraph/CUDA-graph safe).
    Bit-identical semantics to ``_embed_bpe``'s Python sweep:

      * boundary chunk (all bytes in BOUNDARY_BYTES) separates words;
      * non-boundary chunk resolves via ``chunk_to_cb`` (-1 == skip:
        byte_mode key not in codebook);
      * a word = a maximal run of non-boundary chunks, EMITTED only if
        it has >=1 resolved sub-token (empty/all-unresolved runs take
        no slot); per-row emitted words are slotted 0..nObj-1 in
        left-to-right order, later words dropped.

    Everything stays ``[B,T]`` (T = chunk buffer width, static). The
    emitter scatters these into static ``[B*nObj]`` buffers, so the
    target id ``b*nObj+slot`` is the only thing needed -- no dense
    word-id / word-count, hence no host sync.

    Returns (all ``[B,T]``, long/bool): ``sub_cb`` (codebook row, -1
    where not a kept sub-token), ``sub_target`` (``b*nObj+slot``, or
    ``B*nObj`` trash bucket where not kept), ``sub_pos`` (token index,
    for first-sub-token tiebreak), ``keep`` (bool).
    """
    B, T = chunk_ids.shape
    dev = chunk_ids.device
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    valid = pos < tok_count.unsqueeze(1)
    ids = chunk_ids.clamp(min=0)
    is_bnd = tables["is_boundary"][ids] | (~valid)          # pad => sep
    cb = tables["chunk_to_cb"][ids]                          # -1 = skip
    resolved = valid & (~is_bnd) & (cb != -1)                # [B,T]

    # run_key = #boundary chunks strictly before this position (a run
    # of non-boundary chunks between two boundaries shares one key).
    bnd_i = is_bnd.to(torch.int64)
    run_key = torch.cumsum(bnd_i, dim=1) - bnd_i             # [B,T]

    # Per (b, run_key): does the run hold >=1 resolved sub-token?
    has_res = torch.zeros(B, T, dtype=torch.int64, device=dev)
    has_res.scatter_reduce_(
        1, run_key, resolved.to(torch.int64),
        reduce="amax", include_self=True)                    # 0/1 grid

    # Slot for an emitted run = #emitted runs with smaller run_key
    # (exclusive cumsum of the has_res grid over the run-key axis).
    emit_excl = torch.cumsum(has_res, dim=1) - has_res       # [B,T]
    slot = emit_excl.gather(1, run_key)                      # [B,T]
    keep = resolved & (slot < nObj)                          # [B,T] bool

    b_idx = (torch.arange(B, device=dev)
             .unsqueeze(1).expand(B, T))                     # [B,T]
    trash = B * nObj
    sub_target = torch.where(keep, b_idx * nObj + slot,
                             torch.full_like(slot, trash))
    sub_cb = torch.where(keep, cb, torch.full_like(cb, -1))
    return sub_cb, sub_target, pos, keep


def gpu_chunk_ids(byte_buf, best_id, best_len):
    """Greedy left-to-right consumption -> token-major ``chunk_ids``
    [B, N] long (padded with -1) and ``tok_count`` [B] long.

    Sequential by nature (token k+1 starts at cursor + L[cursor]); a
    bounded on-device scan over a ``[B]`` cursor, fixed N iterations
    with masking -- no ``.any()``/``.item()`` host sync. N is the byte
    buffer width; #tokens per row <= N so N iterations always suffice.
    """
    B, N = byte_buf.shape
    dev = byte_buf.device
    chunk_ids = torch.full((B, N), -1, dtype=torch.int64, device=dev)
    cursor = torch.zeros(B, dtype=torch.int64, device=dev)
    tok_count = torch.zeros(B, dtype=torch.int64, device=dev)
    ar = torch.arange(B, device=dev)
    for t in range(N):
        c = cursor.clamp(max=N - 1)
        cur_byte = byte_buf[ar, c]
        active = (cursor < N) & (cur_byte != 0)
        cur_id = best_id[ar, c]
        cur_len = best_len[ar, c]
        chunk_ids[ar, torch.full_like(cursor, t)] = torch.where(
            active, cur_id, chunk_ids[ar, torch.full_like(cursor, t)])
        tok_count = tok_count + active.to(torch.int64)
        cursor = cursor + torch.where(
            active, cur_len, torch.zeros_like(cur_len))
    return chunk_ids, tok_count
