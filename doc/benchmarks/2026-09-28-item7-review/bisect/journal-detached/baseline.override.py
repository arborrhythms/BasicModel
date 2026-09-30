def _tensor_record_operation_values(journal, slot, state, choice):
    """Capture actual operands before fusion and the selected result."""
    buffer = state[0]
    B, K, D = buffer.shape
    selected = buffer.gather(1,
        choice.position.clamp(0, K - 1)[:, None, None].expand(B, 1, D))[:, 0]
    binary = choice.kind == 1
    left = torch.where(binary[:, None], buffer[:, min(1, K - 1)], selected)
    right = torch.where(binary[:, None], buffer[:, 0], torch.zeros_like(selected))
    values = torch.cat((left, right, choice.candidate), -1)
    column = slot.reshape(-1, 1, 1).expand(B, 1, 3 * D).clamp(0, journal.shape[1] - 1)
    values = torch.where(choice.applied[:, None], values, journal.gather(1, column)[:, 0])
    return journal.scatter(1, column, values[:, None]).detach()
