"""A part of a sentence read at one time by the basic level.

A look over the words of one constituent gives the order-zero field of
those words alone; the idea stack (``ConceptualSpace.idea_stack``) reads it
to one idea, as it reads a whole sentence's open field (``IdeaRead``). The
constituent's part of speech is read from that idea by the grammar's
declared argument concepts (``PartsOfSpeech.value_scores``).

System 2 teaches it. At a kept sentence's close each constituent of the
kept reading whose words are one run of the sentence, of two to ``LENGTH``
words, gives the targets: its composed point (the compose's recorded value
at that node) for the read idea, and the argument seat it filled in its
parent operation for the read part of speech. The field and the targets are
detached, so the error reaches the idea stack and nothing else.

The grammar's option (``fragmentBias``, the probability of taking the read,
zero by default): at a look, the run of unread words in the look's extent
that begins at the word taken, of two to ``LENGTH`` words, may be taken as
one leaf whose value is the read idea and whose part of speech is the read
one; the other words of the run are read with it and do not arrive. No form,
row or symbol is made for the run. With the probability at zero nothing here
runs.
"""
import torch

LENGTH = 3


def bias(model):
    """The probability of taking a read part, ``fragmentBias``."""
    layer = model._stm_reducer()
    return 0. if layer is None else float(getattr(layer, 'fragment_bias', 0.))


def knowing(owner, bank, words):
    """The order-zero field ``[B, n0, 2]`` of a look over the word slots
    ``words`` ``[B, W]`` of ``bank``: each occurrence present at its own activity
    where its word is in ``words``, and absent elsewhere. Nothing is
    published and nothing is kept."""
    from BasicPercepts import reading_scope
    held = (reading_scope(bank, words.shape[1]) & words[:, :, None]).any(1)
    activity = torch.where(held[..., None], bank.activity, torch.zeros_like(bank.activity))
    with torch.no_grad():
        evidence = owner.cs_read_bank(bank, activity, publish=False)
    return evidence.amax(2).transpose(0, 1)[:, :owner._order_caps()[0]].detach()


def read(owner, field):
    """The idea ``[B, D]`` read from an order-zero field ``[B, n0, 2]``."""
    return owner.idea_stack(field.transpose(-1, -2))


def classes(language, idea):
    """The part of speech read from ``idea``: its coverage of each declared
    argument concept, ``[B, A]``."""
    return language.parts_of_speech.value_scores(idea)


def constituents(language, program, slots, length=LENGTH):
    """The constituents of a kept reading that a look can hold: ``(action,
    words, seat)`` for each operation whose leaves are the word slots
    ``words``, one run of the sentence of two to ``length`` words, with
    ``seat`` the argument it filled in its parent operation (None at a
    root). ``slots`` gives each leaf's word slot."""
    from PartsOfSpeech import operation_name
    index = program.action_index
    under = {}
    found = []
    for at, (kind, local, word) in enumerate(index.steps):
        if kind == 0:
            under[at] = (slots[word],) if 0 <= word < len(slots) else ()
            continue
        if kind == 3:
            continue
        under[at] = tuple(sorted(w for child in index.operands[at] for w in under.get(child, ())))
        words = under[at]
        if not 2 <= len(words) <= length or len(set(words)) != len(words):
            continue
        if words[-1] - words[0] != len(words) - 1:
            continue
        parent = index.parents[at]
        if parent is not None and index.steps[parent][0] in (1, 2) and tuple(sorted(
                w for child in index.operands[parent] for w in under.get(child, ()))) == words:
            # A unary operation over the same words: the constituent is the
            # last operation over them, the one its parent reads.
            continue
        seat = None
        if parent is not None and index.steps[parent][0] in (1, 2):
            pkind, plocal, _ = index.steps[parent]
            rule = (language._compose_unary_rules if pkind == 2 else language._compose_binary_rules)[plocal]
            name = f'{operation_name(rule)}_argin{index.operands[parent].index(at) + 1}'
            seat = name if name in language.parts_of_speech.index else None
        found.append((at, words, seat))
    return found


def complete(model, observations, selected, present, registry, sid, looks=()):
    """Teach the read of each kept constituent at the sentence's close. A
    kept path that took a read has no composition of those words to teach
    by, so its stream is not taught."""
    if bias(model) <= 0.:
        return
    owner = model._concept_owner()
    if getattr(owner, 'idea_stack', None) is None:
        return
    bank = model.__dict__.get('_sentence_banks', {}).get(sid)
    orders = getattr(model, '_sentence_read_order', None)
    if bank is None or orders is None:
        return
    language = model.languageSpace
    B = len(present)
    W = int(bank.token.max()) + 1 if bank.token.numel() else 0
    idea_error = [[] for _ in range(B)]
    class_error = [[] for _ in range(B)]
    for b in present.nonzero().flatten().tolist():
        path = int(selected[b])
        walked = looks[path] if path < len(looks) else None
        if walked is not None and any(bool(d.get('fragment_taking', torch.zeros(B, dtype=torch.bool))[b])
                                      for d in walked.decisions):
            continue
        program = observations[path]['entries'][b]
        if program is None or program.operation_values is None or b >= len(orders):
            continue
        slots = list(orders[b])
        if len(slots) != len(program.leaves):
            continue
        for at, words, seat in constituents(language, program, slots):
            mask = torch.zeros(B, max(W, words[-1] + 1), dtype=torch.bool, device=bank.token.device)
            mask[b, list(words)] = True
            idea = read(owner, knowing(owner, bank, mask))[b]
            target = program.operation_values[at, 2].detach().to(idea)
            idea_error[b].append((idea - target).square().mean())
            if seat is not None:
                scores = classes(language, idea[None])[0]
                wanted = torch.zeros_like(scores)
                wanted[language.parts_of_speech.index[seat]] = 1.
                class_error[b].append((scores - wanted).square().mean())
    for name, errors in (('expectation.fragment', idea_error), ('expectation.fragment_class', class_error)):
        held = torch.tensor([bool(e) for e in errors], device=present.device)
        if not bool(held.any()):
            continue
        like = next(e[0] for e in errors if e)
        error = torch.stack([torch.stack(e).mean() if e else like.new_zeros(()) for e in errors])
        registry.error(name, error, 1., mask=held, category='expectation')


def run_at(state, extent, index, rows, length=LENGTH):
    """The run of unread words ``[B, W]`` in a look's extent that begins at the
    word taken, ``index``, and goes on through the next slots while each is
    in the extent, unread and a known word, up to ``length`` words."""
    B, W = state.known.shape
    width = min(W, extent.shape[1])
    extent = extent[:, :width]
    open_ = extent & ~state.accepted[:, :width] & state.known[:, :width]
    if width < 2 or length < 2:
        return torch.zeros_like(state.known)
    slots = torch.arange(width, device=index.device)
    start = index[:, None]
    blocked = (slots > start) & ~open_
    end = torch.where(blocked, slots, width).amin(-1, keepdim=True).minimum(start + length)
    eligible = rows[:, None] & (start >= 0) & (end > start + 1)
    run = eligible & (slots >= start) & (slots < end)
    return torch.nn.functional.pad(run, (0, W - width))


def take(model, state, bank, extent, index, rows, *, prior=None):
    """Whether each stream takes the read of the run beginning at ``index``.

    Returns None where nothing is taken; otherwise the run ``[B, W]``, the
    streams taking it ``[B]``, the read idea ``[B, D]`` and the read part of
    speech ``[B, A]``. ``prior`` replays a recorded draw."""
    p = bias(model)
    if p <= 0. or bank is None:
        return None
    owner = model._concept_owner()
    if getattr(owner, 'idea_stack', None) is None:
        return None
    run = run_at(state, extent, index, rows)
    eligible = rows & run.any(-1)
    if prior is not None:
        taking = eligible & prior
    else:
        taking = eligible & (torch.rand(len(rows), device=rows.device) < p)
    if not bool(taking.any()):
        return dict(run=run, taking=taking, idea=None, classes=None)
    run = run & taking[:, None]
    idea = read(owner, knowing(owner, bank, run))
    return dict(run=run, taking=taking, idea=idea, classes=classes(model.languageSpace, idea).detach())


def as_leaf(model, payload, index, rows):
    """The payload of a word whose arrival is a taken read: its value is the
    read idea, and its part of speech the read one in the word table. Returns
    the payload and a function that puts the word table back."""
    held = model.__dict__.get('_fragment_leaf')
    if held is None or held['idea'] is None:
        return payload, None
    taking = rows & held['taking'] & (held['index'] == index)
    if not bool(taking.any()):
        return payload, None
    event, atom = payload[0], payload[6]
    idea = held['idea'].detach() if not torch.is_grad_enabled() else held['idea']
    if event.shape[-1] == idea.shape[-1]:
        event = torch.where(taking[:, None, None], idea[:, None].to(event), event)
    if atom.shape[-1] == idea.shape[-1]:
        atom = torch.where(taking[:, None], idea.to(atom), atom)
    payload = (event, *payload[1:6], atom, *payload[7:])
    table = getattr(model, '_occupation_word_table', None)
    restore = None
    if torch.is_tensor(table) and table.shape[-1] == held['classes'].shape[-1]:
        slot = int(index)
        saved = table[:, slot].clone()
        with torch.no_grad():
            table[:, slot] = torch.where(taking[:, None], held['classes'].to(table), table[:, slot])

        def restore():
            with torch.no_grad():
                table[:, slot] = saved
    return payload, restore
