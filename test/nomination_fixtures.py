"""A word read until MTM nominates it (sigma-pi §21, rule 20).

A form becomes a word only on its ``concept_mint_count``-th reading. A
fixture that needs a word reads its form that many times.
"""


def recurrent_word(interpret, *args, **kwargs):
    """``InterpretLayer.lookup_word`` read until MTM nominates the form."""
    owner = interpret.owner
    word = None
    for reading in range(owner.concept_mint_count):
        word = interpret.lookup_word(*args, reading=('fixture', reading), **kwargs)
    return word


def nominated_forms(model, texts):
    """Admit the forms of ``texts`` and read each until MTM nominates it:
    ``(raw, word)`` for each lexical form."""
    from Corpus import admit_training_forms
    owner = model._concept_owner()
    result = []
    for raw, parts in admit_training_forms(model, texts):
        wholes = model.wholeSpace.property_rows_for_bytes(raw)
        result.append((raw, recurrent_word(owner.interpret, parts, wholes, form=raw)))
    return tuple(result)


def readings_before_nomination(model, read):
    """The model's own readings that come before the one in which its forms
    are found: ``read()`` (one reading of the text) ``concept_mint_count - 1``
    times, each closed by ``model.End()``. The caller's next reading of the
    same text is the ``concept_mint_count``-th, in which MTM nominates the
    forms read in each one."""
    import torch
    for _ in range(model._concept_owner().concept_mint_count - 1):
        with torch.no_grad():
            read()
        model.End()


def recurrent_interpretation(owner, *args, **kwargs):
    """``ConceptualSpace.interpret_word`` read until MTM nominates the form."""
    last = kwargs.pop('occurrence', ('fixture', owner.concept_mint_count, 0))
    result = None
    for reading in range(owner.concept_mint_count):
        occurrence = last if reading == owner.concept_mint_count - 1 else (('fixture', reading), 0, 0)
        result = owner.interpret_word(*args, occurrence=occurrence, **kwargs)
    return result
