def resolve_operand(value, identity, position, *, mode, bank, live, active, forced_reference=None):
    """A hard reference proposal before its numerical grammar operation.

    Only the existing predictor anchors and earlier occupied constituents are
    candidates. The caller masks unavailable proposals before the global
    operation softmax; an unselected pronoun proposal cannot reject a reading.
    """
    live_values, live_ids, live_orders, live_relations, live_positions = live
    B, P, D = value.shape
    C = bank.ids.shape[1]
    K = live_ids.shape[1]
    own_valid = (identity > 0) & (mode != 'pronoun')
    held = bank.valid & bank.predicted[:, None]
    earlier = (live_positions[:, None, :] < position[:, :, None])
    earlier &= (live_ids[:, None, :] > 0) & (live_orders[:, None, :] == 1)
    ids = torch.cat((identity[:, :, None], bank.ids[:, None].expand(B, P, C),
                     live_ids[:, None].expand(B, P, K)), -1)
    values = torch.cat((value[:, :, None], bank.values[:, None].expand(B, P, C, D),
                        live_values[:, None].expand(B, P, K, D)), -2)
    valid = torch.cat((own_valid[:, :, None], held[:, None].expand(B, P, C), earlier), -1)
    relations = torch.cat((torch.zeros_like(own_valid[:, :, None]),
                          bank.relations[:, None].expand(B, P, C),
                          live_relations[:, None].expand(B, P, K)), -1)
    # The first occurrence of an address owns its candidate mass.
    size = ids.shape[-1]
    preceding = torch.arange(size, device=ids.device)[
        None, :] < torch.arange(size, device=ids.device)[:, None]
    duplicate = ((ids[..., :, None] == ids[..., None, :]) & preceding & valid[..., None, :]).any(-1)
    valid = valid & ~duplicate
    available = valid.any(-1)
    # An unavailable proposal receives a harmless local value but is masked
    # out of the operation chooser. Keep the softmax finite for all rows.
    safe_valid = valid | (~available[:, :, None] & (torch.arange(size, device=ids.device) == 0))
    query = torch.where(bank.predicted[:, None, None], bank.query[:, None, :], value)
    logits = F.cosine_similarity(
        values, query[:, :, None, :], dim=-1).masked_fill(~safe_valid, -torch.inf)
    probabilities = logits.softmax(-1)
    selected = probabilities.argmax(-1)
    if forced_reference is not None:
        forced = torch.as_tensor(forced_reference, device=ids.device)
        permitted = valid & (ids == forced[..., None])
        if not torch.compiler.is_compiling() and not bool(permitted.any(-1).all()):
            raise ValueError('forced identity is outside the bounded candidate set')
        torch._assert_async(permitted.any(-1).all(),
                            'forced identity is outside the bounded candidate set')
        selected = permitted.to(torch.long).argmax(-1)
    point = values.gather(2, selected[:, :, None, None].expand(B, P, 1, D)).squeeze(2)
    probability = probabilities.gather(2, selected[:, :, None]).squeeze(2)
    resolved = point
    refs = ids.gather(2, selected[:, :, None]).squeeze(2)
    relative = relations.gather(2, selected[:, :, None]).squeeze(2)
    used = active & available
    return (torch.where(used[:, :, None], resolved, value),
            torch.where(used, refs, identity), relative & used, available | ~active)
