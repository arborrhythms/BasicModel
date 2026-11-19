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
