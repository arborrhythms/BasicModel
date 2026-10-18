"""The scoped constituent experiment owns exactly its declared paired updates."""
from contextlib import contextmanager


@contextmanager
def paired_updates_only(model, optimizer):
    """Disable the optional batch-end step, recording every attempted update."""
    native_train, native_step = model._sentence_train_step, optimizer.step
    counts = dict(paired=0, batch_end_disabled=0)
    inside = False
    def train(*args, **kwargs):
        nonlocal inside
        inside = True
        try:
            return native_train(*args, **kwargs)
        finally:
            inside = False
    def step(*args, **kwargs):
        if not inside:
            counts['batch_end_disabled'] += 1
            optimizer.zero_grad()
            return None
        result = native_step(*args, **kwargs)
        counts['paired'] += 1
        return result
    model._sentence_train_step, optimizer.step = train, step
    try:
        yield counts
    finally:
        model._sentence_train_step, optimizer.step = native_train, native_step
